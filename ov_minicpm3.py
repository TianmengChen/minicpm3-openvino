import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import AutoConfig
from typing import List
import logging as log
from pathlib import Path
from transformers.generation import GenerationConfig, GenerationMixin
import numpy as np
from openvino.runtime import opset13
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
import PIL
from PIL import Image

from typing import Optional, Tuple, List, Union

import openvino as ov
from openvino.runtime import Core, Type
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
from openvino.runtime import opset10 as ops
from openvino.preprocess import PrePostProcessor
import nncf

import time
import warnings

# from conversation import get_conv_template

def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [
        key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())
    ]
    key_value_output_names = [
        key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())
    ]
    not_kv_inputs = [
        input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())
    ]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1
    
    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )   

class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            print("root: ", root)
            if root is None:
                return False
            root_output = matcher.get_match_value()
            print("root_output", root_output)
            root_name = root.get_friendly_name()
            if (len(root.get_output_partial_shape(0)) == 3):
                print(f"Find target root node name: {root_name}")
                parent = root.input_value(0).get_node()
                print(f"Find target parent node name: {parent.get_friendly_name()}")
                grand_parent = parent.input_value(0).get_node()
                print(f"Find grandparent node name: {grand_parent.get_friendly_name()}")
                grand_parent_output = parent.input(0).get_source_output()
                print("grand_parent_output: ", grand_parent_output)
                consumers = grand_parent_output.get_target_inputs()
                
                print(f"consumers: {consumers}")
                print("Original reshape node output shape:", grand_parent_output.get_partial_shape())
                start = np.array([0, -1, 0], dtype=np.int32)
                stop = np.array([1, -2, 2560], dtype=np.int32)
                step = np.array([1, -1, 1], dtype=np.int32)
                axes = np.array([0, 1, 2], dtype=np.int32)
                slice = ops.slice(grand_parent, start, stop, step, axes, name="inserted_slice")
                print("After insert slice node, output shape:", slice.output(0).get_partial_shape())

                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)
                                
                return True

        self.register_matcher(Matcher(param,"InsertSlice"), callback)



class LlmStatefulModel():
    def __init__(
        self,
        model=None,
        tokenizer=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
        int4_compress=False,
    ):
        self.name = "MiniCPM3-4B LLM Model"
        self.model = model
        self.tokenizer = tokenizer
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.int4_compress = int4_compress
        self.inputs_dict = {}

    def get_model(self):
        return self.model

    def get_input_names(self):
        inputs = ['attention_mask', 'position_ids']
        for idx in range(len(self.model.model.layers)):
            inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        inputs.append('inputs_embeds')
        return inputs

    def get_output_names(self):
        outputs = ['logits']
        for idx in range(len(self.model.model.layers)):
            outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
            pass
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f'tokenizer loading failed with {e}')

    def convert_sdpa_ov(self):
        llm_model = self.get_model()        
        with torch.no_grad():
            pkv = llm_model.model(input_ids=torch.tensor([[ 73441,  3060,     5,  5147, 59367, 59411,  3083, 59350, 20349,    66,
                73440, 59320,     5, 73441, 16434,     5]]).to(torch.int64),
                                position_ids=torch.tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]]).to(torch.int64),
                                attention_mask=torch.ones((1, 16), dtype=torch.int64), use_cache=True, return_dict=False)[1]

        attention_mask = torch.ones(1, 17)
        import numpy as np
        position_ids = torch.tensor([[17-1]])

        llm_model.config.torchscript = True
        ov_model = ov.convert_model(
            llm_model,
            example_input={
                "inputs_embeds":  torch.randn(( 1, 1, 2560), dtype=torch.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": pkv,
             },
        )
        print("stateful model inputs: ", ov_model.inputs)
        # breakpoint()
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        # patch_stateful(ov_model)
        # manager = Manager()
        # manager.register_pass(InsertSlice())
        # manager.run_passes(ov_model)

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_pkv.xml"))
        self.save_tokenizer(self.tokenizer, self.ov_model_path)
        self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/llm_stateful_int4.xml"))
    

class LlmModel(LlmStatefulModel):
    def __init__(
        self,
        model=None,
        tokenizer=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
        int4_compress=False,
    ):
        super().__init__(model, tokenizer, ov_model_path, device, fp16, int4_compress)

    def get_input_names(self):
        inputs = ['attention_mask', 'position_ids', 'inputs_embeds']
        return inputs

    def convert_sdpa_ov(self):
        llm_model = self.get_model()      
        attention_mask = torch.ones(1, 16)

        llm_input = torch.rand((1, 16, 2560), dtype=torch.float32)
        pkv = None
        # pkv = llm_model(inputs_embeds=llm_input, attention_mask=attention_mask, use_cache=True, return_dict=False)[1]
        # breakpoint()
        attention_mask = torch.ones(1, 16)
        import numpy as np
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])

        llm_model.config.torchscript = True
        ov_model = ov.convert_model(
            llm_model,
            example_input={
                "inputs_embeds":  llm_input,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
             },
        )
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})
        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_nopkv.xml"))

class LlmEmbdModel():
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "MiniCPM3-4B LLM Embd Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.model.embed_tokens

    def get_input_names(self):
        inputs = ['input_ids']
        return inputs

    def get_output_names(self):
        outputs = ['inputs_embeds']
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
            pass

    def convert_sdpa_ov(self):
        embd_model = self.get_model()        

        input_ids = torch.randint(0, 32020, ( 1, 3408))

        ov_model = ov.convert_model(
            embd_model,
            example_input={
                "input":  input_ids,
             },
        )
        # breakpoint()
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_embd.xml"))


class MiniCPM3_OV:
    def __init__(self, pretrained_model_path=None, model=None, tokenizer=None, ov_model_path='/tmp/MiniCPM3_ov/', device='CPU', llm_int4_compress=False):

        if model is None and pretrained_model_path:        
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_path,
                trust_remote_code=True,
                _attn_implementation='sdpa',
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path, 
                trust_remote_code=True
            )
        elif model and tokenizer and pretrained_model_path is None:
            self.model = model
            self.tokenizer = tokenizer

        self.int4_compress = llm_int4_compress

        self.llm_embed_model = LlmEmbdModel(model=self.model, ov_model_path=ov_model_path, device=device)
        self.llm_stateful_model = LlmStatefulModel(model=self.model, tokenizer= self.tokenizer, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress)
        self.llm_nopkv_model = LlmModel(model=self.model, tokenizer= self.tokenizer, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress)

    def export_vision_to_ov(self):
        self.llm_embed_model.convert_sdpa_ov()
        self.llm_nopkv_model.convert_sdpa_ov()
        self.llm_stateful_model.convert_sdpa_ov()
        

class OVMiniCPM3ForCausalLM(GenerationMixin):
    def __init__(
        self,
        core=None,
        ov_model_path=None,
        device='CPU',
        llm_int4_compress=False, 
        vision_int8_quant=False, 
        llm_int8_quant=False,
        llm_infer_list=[],
    ):
        self.ov_model_path = ov_model_path
        self.core = core
        self.ov_device = device
        self.llm_int4_compress = llm_int4_compress
        self.vision_int8_quant = vision_int8_quant
        self.llm_int8_quant = llm_int8_quant

        ov_config = {
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": "128",  #32
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "CACHE_DIR": "",
        }

        if llm_int4_compress:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/llm_stateful_int4.xml"))
        else:
            self.llm_pkv_model = core.read_model(Path(f"{ov_model_path}/llm_pkv.xml"))
            self.llm_nopkv_model = core.read_model(Path(f"{ov_model_path}/llm_nopkv.xml"))
            
        if llm_int8_quant:
            self.llm_pkv_compiled_model = core.compile_model(self.llm_pkv_model, device, config = ov_config)
            self.llm_nopkv_compiled_model = core.compile_model(self.llm_nopkv_model, device, config = ov_config)
        else:
            self.llm_pkv_compiled_model = core.compile_model(self.llm_pkv_model, device)
            self.llm_nopkv_compiled_model = core.compile_model(self.llm_nopkv_model, device)
            
        self.llm_pkv_request = self.llm_pkv_compiled_model.create_infer_request()
        self.llm_nopkv_request = self.llm_nopkv_compiled_model.create_infer_request()

        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.llm_pkv_model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.llm_pkv_model.outputs)}
        self.key_value_input_names = [key for key in list(self.input_names) if key not in ["beam_idx", "inputs_embeds", "attention_mask", "position_ids"]]
        self.key_value_output_names = [key for key in list(self.output_names)[1:]]
        self.stateful = len(self.key_value_input_names) == 0
        # self.compiled_model = core.compile_model(self.model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})

        self.config = AutoConfig.from_pretrained(ov_model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.device = torch.device("cpu")
        self.next_beam_idx = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = None
        self.main_input_name = "input_ids"
        self._supports_cache_class = False

        self.llm_embd = core.read_model(Path(f"{ov_model_path}/llm_embd.xml"))
        self.llm_embd_compiled_model = core.compile_model(self.llm_embd, device)
        self.llm_embd_request = self.llm_embd_compiled_model.create_infer_request()
        
        self.tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)

        self.llm_infer_list = llm_infer_list


    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True
    
    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def llm_embd_run(self, input_ids):
        llm_embd_inputs = {}
        llm_embd_inputs['input_ids'] = input_ids

        self.llm_embd_request.start_async(llm_embd_inputs, share_inputs=True)
        self.llm_embd_request.wait()

        return torch.from_numpy(self.llm_embd_request.get_tensor("inputs_embeds").data)*12

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""
        if past_key_values is not None:
            inputs_dict = {}
            #deal with pkv
            # print('------------new round------------')
            for idx in range(62):
                # print('shape ', past_key_values[2*idx].shape)
                # print('shape ', past_key_values[2*idx+1].shape)
                inputs_dict[f"past_key_values.{idx}.key"] = past_key_values[2*idx]
                inputs_dict[f"past_key_values.{idx}.value"] = past_key_values[2*idx+1]
            # print("input_ids: ", input_ids)
            inputs_embeds = self.llm_embd_run(input_ids)
            inputs_dict['inputs_embeds'] = inputs_embeds

            inputs_dict["attention_mask"] = attention_mask
            inputs_dict["position_ids"] = position_ids

            batch_size = inputs_embeds.shape[0]
            # if "beam_idx" in self.input_names:
            #     inputs_dict["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

            # print('attention_mask: ', inputs_dict['attention_mask'].shape)
            # print('position_ids: ', inputs_dict['position_ids'])
            # print('inputs_embeds: ', inputs_dict['inputs_embeds'].shape)
            # print("beam_idx: ", inputs_dict["beam_idx"])
            start = time.perf_counter()
            self.llm_pkv_request.start_async(inputs_dict, share_inputs=True)
            self.llm_pkv_request.wait()
            end = time.perf_counter()

            generation_time = (end - start) * 1000
            self.llm_infer_list.append(generation_time)

            self.past_len += inputs_dict["inputs_embeds"].shape[1]
            logits=torch.from_numpy(self.llm_pkv_request.get_tensor("logits").data)

            #deal with pkv
            past_key_values=[]
            
            for index in range(1,125):
                past_key_values.append(self.llm_pkv_request.get_output_tensor(index).data)

            # print('logits: ', self.llm_request.get_tensor("logits").data)
            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=None,
                attentions=None,
            )   
        else:
            # print('------------first round------------')
            inputs_dict = {}

            self.past_len = 0
            self.llm_nopkv_request.reset_state()
            self.llm_pkv_request.reset_state()
            
            inputs_dict['inputs_embeds'] = inputs_embeds

            inputs_dict["attention_mask"] = attention_mask
            inputs_dict["position_ids"] = position_ids

            batch_size = inputs_embeds.shape[0]

            # print('attention_mask: ', inputs_dict['attention_mask'].shape)
            # print('position_ids: ', inputs_dict['position_ids'])
            # print('inputs_embeds: ', inputs_dict['inputs_embeds'].shape)
            start = time.perf_counter()
            self.llm_nopkv_request.start_async(inputs_dict, share_inputs=True)
            self.llm_nopkv_request.wait()
            end = time.perf_counter()

            generation_time = (end - start) * 1000
            self.llm_infer_list.append(generation_time)

            # past_key_values = ((),)
            self.past_len += inputs_dict["inputs_embeds"].shape[1]

            logits = torch.from_numpy(self.llm_nopkv_request.get_tensor("logits").data)
            #deal with pkv
            past_key_values=[]
            for index in range(1,125):
                # print("shape: ", self.llm_nopkv_request.get_output_tensor(index).get_shape())
                past_key_values.append(self.llm_nopkv_request.get_output_tensor(index).data)
            
            # print('logits: ', self.llm_request.get_tensor("logits").data)
            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=None,
                attentions=None,
            )   

    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            cache_length = past_length = self.past_len
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - self.past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif self.past_len < input_ids.shape[1]:
                input_ids = input_ids[:, self.past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]
        else:
            self.llm_infer_list.clear()

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:    
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def get_input_embeds(self, input_ids=None):
        input_embeds = self.llm_embd_run(input_ids)
        
        return input_embeds
    
    # def chat(self, pixel_values, question, generation_config, history=None, return_history=False,
    #          num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
    #          verbose=False):
        
    #     if history is None and pixel_values is not None and '<image>' not in question:
    #         question = '<image>\n' + question

    #     if num_patches_list is None:
    #         num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    #     assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    #     img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    #     self.img_context_token_id = img_context_token_id

    #     # template = get_conv_template("phi3-chat")
    #     template.system_message = self.system_message
    #     eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)

    #     history = [] if history is None else history
    #     for (old_question, old_answer) in history:
    #         template.append_message(template.roles[0], old_question)
    #         template.append_message(template.roles[1], old_answer)
    #     template.append_message(template.roles[0], question)
    #     template.append_message(template.roles[1], None)
    #     query = template.get_prompt()

    #     if verbose and pixel_values is not None:
    #         image_bs = pixel_values.shape[0]
    #         print(f'dynamic ViT batch size: {image_bs}')

    #     for num_patches in num_patches_list:
    #         image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
    #         query = query.replace('<image>', image_tokens, 1)

    #     model_inputs = self.tokenizer(query, return_tensors='pt')
    #     input_ids = model_inputs['input_ids']
    #     attention_mask = model_inputs['attention_mask']
    #     generation_config['eos_token_id'] = eos_token_id

    #     inputs_emds = self.get_input_embeds(pixel_values, input_ids)

    #     generation_output = self.generate(
    #         inputs_embeds=inputs_emds,
    #         attention_mask=attention_mask,
    #         **generation_config
    #     )
    #     response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    #     response = response.split(template.sep)[0].strip()
    #     history.append((question, response))
    #     if return_history:
    #         return response, history
    #     else:
    #         query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
    #         query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
    #         if verbose:
    #             print(query_to_print, response)
    #         return response

    

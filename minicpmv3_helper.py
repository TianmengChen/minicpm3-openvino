import torch
from threading import Thread
from copy import deepcopy
import shutil
import json
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoTokenizer
from transformers.generation import GenerationMixin
from transformers import AutoConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from pathlib import Path
from huggingface_hub import snapshot_download
import types
from typing import Optional, Tuple, List, Union
from openvino.runtime import opset13
import openvino as ov
import numpy as np
import gc
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
import time
from transformers.modeling_utils import PreTrainedModel
from optimum.exporters.openvino.stateful import patch_stateful

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()
    
def _wrapper_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPMForCausalLM

        >>> model = MiniCPMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        logits = logits.float()

        output = (logits,) + outputs[1:]
        return output


def convert_minicpmv3_model(model_id, output_dir, quantization_config):
    model_name = Path(model_id).name
    output_dir = Path(output_dir)

    lang_model_path = output_dir / "openvino_model.xml"

    if all(
        [
            lang_model_path.exists(),
            # image_embed_path.exists(),
            # embed_token_path.exists(),
        ]
    ):
        print(f"✅ {model_name} model already converted. You can find results in {output_dir}")
        return
    print(f"⌛ {model_name} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32, _attn_implementation="sdpa")
    model.config.save_pretrained(output_dir)
    with torch.no_grad():
        pkv = model.model(input_ids=torch.randint(0, 1000, size=(1, 16)).to(torch.int64),
                          position_ids=torch.tensor([[[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]]]]).to(torch.int64),
                        attention_mask=torch.ones((1, 16), dtype=torch.int64), use_cache=True, return_dict=False)[1]


    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("⌛ Convert to OpenVINO IR")
    model.forward = types.MethodType(_wrapper_forward, model)
    model.eval()

    # hidden_size = model.model.config.hidden_size
    # num_pkv = model.model.config.num_hidden_layers
    # pkv_shape = (2, model.model.config.num_key_value_heads, 2, hidden_size // model.model.config.num_attention_heads)
    input_ids=torch.randint(0, 1000, size=(1, 1)).to(torch.int64)
    inputs_embeds = torch.randn(( 1, 1, 2560), dtype=torch.float32)
    attention_mask = torch.ones((1, 17), dtype=torch.int64)
    position_ids = torch.tensor([[16]]).to(torch.int64)
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits"]

    for i in range(len(pkv)):
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])

    # input_names.append("inputs_embeds")

    example_input = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": pkv}
    # example_input = {"attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": pkv, "inputs_embeds": inputs_embeds}
    model.config.torchscript = True
    # breakpoint()
    ov_model = ov.convert_model(model, example_input=example_input)

    for out, out_name in zip(ov_model.outputs, output_names):
        out.get_tensor().set_names({out_name})

    for inp, inp_name in zip(ov_model.inputs, input_names):
        inp.get_tensor().set_names({inp_name})

    patch_stateful(config=model.config,ov_model=ov_model)
    ov.save_model(ov_model, lang_model_path)
    del ov_model

    cleanup_torchscript_cache()
    gc.collect()
    print("✅ Language model successfully converted")


# class OVMINICPM3(GenerationMixin):
#     d
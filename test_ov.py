import argparse
import openvino as ov
from pathlib import Path
from ov_minicpm3 import OVMiniCPM3ForCausalLM, MiniCPM3_OV
from transformers import TextStreamer
import time
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Export MiniCPM3-4B Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=False, help="model_id or directory for loading")
    parser.add_argument("-ov", "--ov_ir_dir", required=True, help="output directory for saving model")
    parser.add_argument('-d', '--device', default='CPU', help='inference device')
    parser.add_argument('-p', '--prompt', default="Describe this image.", help='prompt')
    parser.add_argument('-max', '--max_new_tokens', default=512, help='max_new_tokens')
    parser.add_argument('-llm_int4_com', '--llm_int4_compress', action="store_true", help='llm int4 weights compress')
    parser.add_argument('-llm_int8_quant', '--llm_int8_quant', action="store_true", help='llm int8 weights quantize')
    parser.add_argument('-convert_model_only', '--convert_model_only', action="store_true", help='convert model to ov only, do not do inference test')

    args = parser.parse_args()
    model_id = args.model_id
    ov_model_path = args.ov_ir_dir
    device = args.device
    max_new_tokens = int(args.max_new_tokens)
    question = args.prompt
    llm_int4_compress = args.llm_int4_compress
    llm_int8_quant = args.llm_int8_quant
    convert_model_only=args.convert_model_only

    if not Path(ov_model_path).exists():
        minicpm3_ov = MiniCPM3_OV(pretrained_model_path=model_id, ov_model_path=ov_model_path, device=device, llm_int4_compress=llm_int4_compress)
        minicpm3_ov.export_vision_to_ov()
        del minicpm3_ov.model
        del minicpm3_ov.tokenizer
        del minicpm3_ov
    elif Path(ov_model_path).exists() and llm_int4_compress is True and not Path(f"{ov_model_path}/llm_stateful_int4.xml").exists():
        minicpm3_ov = MiniCPM3_OV(pretrained_model_path=model_id, ov_model_path=ov_model_path, device=device, llm_int4_compress=llm_int4_compress)
        minicpm3_ov.export_vision_to_ov()
        del minicpm3_ov.model
        del minicpm3_ov.tokenizer
        del minicpm3_ov
    
    if not convert_model_only:
        llm_infer_list = []
        core = ov.Core()
        minicpm3_model = OVMiniCPM3ForCausalLM(core=core, ov_model_path=ov_model_path, device=device, llm_int4_compress=llm_int4_compress, llm_int8_quant=llm_int8_quant, llm_infer_list=llm_infer_list)

        version = ov.get_version()
        print("OpenVINO version \n", version)
        print('\n')

        generation_config = {
                "bos_token_id": minicpm3_model.tokenizer.bos_token_id,
                "pad_token_id": minicpm3_model.tokenizer.bos_token_id,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
        messages = [
            {"role": "user", "content": "你是谁"},
        ]
        print(messages[0]['content'])
        input_ids = minicpm3_model.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        # print("input_ids: ", input_ids)

        inputs_embeds = minicpm3_model.get_input_embeds(input_ids=input_ids)
        model_outputs = minicpm3_model.generate(
            inputs_embeds=inputs_embeds,
            **generation_config,
        )
        # print("model_outputs: ", model_outputs)

        responses = minicpm3_model.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)[0]
        print(responses)


        for i in range(4):
            model_outputs = minicpm3_model.generate(
                inputs_embeds=inputs_embeds,
                **generation_config,
                )
            # responses = minicpm3_model.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)[0]
            # print(responses)

        ## i= 0 is warming up
        if i != 0:
            print("\n")
            if len(llm_infer_list) > 1:
                avg_token = sum(llm_infer_list[1:]) / (len(llm_infer_list) - 1)
                print(f"LLM Model First token latency: {llm_infer_list[0]:.2f} ms, Output len: {len(llm_infer_list) - 1}, Avage token latency: {avg_token:.2f} ms")

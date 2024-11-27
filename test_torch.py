from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "MiniCPM3-4B"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [
    {"role": "user", "content": "推荐5个北京的景点。"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

with torch.no_grad():
    # model_outputs = model.forward(**model_inputs)
    model_outputs = model.generate(
        model_inputs,
        max_new_tokens=128,
        return_dict=False
    )

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)

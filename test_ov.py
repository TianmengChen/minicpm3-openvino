import openvino as ov
core = ov.Core()
ov_model=core.read_model("MiniCPM3-4B-ov/openvino_model.xml")


from transformers import AutoTokenizer, AutoConfig
import torch
from optimum.intel import OVModelForCausalLM

model_dir = "MiniCPM3-4B-ov"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = OVModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True, _attn_implementation="sdpa")

messages = [
    {"role": "user", "content": "推荐5个北京的景点。"},
]
inputs = tokenizer.apply_chat_template(messages, 
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True,
                                       add_special_tokens=False,
                                       trust_remote_code=False)

print("====Compiling model====")
ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device="CPU",
    ov_config=None,
    config=AutoConfig.from_pretrained(model_dir),
    trust_remote_code=False,
)
print("====Compiled model====")
ans=ov_model.generate(**inputs)
print(tokenizer.batch_decode(ans))
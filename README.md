## Update Notes
### 2024/11/28
1. MiniCPM3 model supports using openvino to accelerate the inference process. Currently only verified on Linux system and only tested on CPU platform.

## Running Guide
### Installation


```bash
git clone https://github.com/TianmengChen/minicpm3-openvino.git
pip install --pre -U openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
pip install nncf
pip install transformers==4.44.2
pip install torch
pip install torchvision

```
### Convert MiniCPM3 model to OpenVINO™ IR(Intermediate Representation) and testing (Intel(R) Xeon(R) Gold 6252N CPU @ 2.30GHz):
#### FP16
```shell
cd minicpm3-openvino
#linux
python3 test_ov.py -m /path/to/MiniCPM3-4B -ov MiniCPM3-4B-ov 

#output
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
OpenVINO version 
 2025.0.0-17426-287ab9883ac


你是谁
你好，我是MiniCPM系列模型，由面壁智能和OpenBMB开源社区开发。详细信息请访问 https://github.com/OpenBMB/


LLM Model First token latency: 185.88 ms, Output len: 34, Avage token latency: 119.80 ms
```
#### INT4 compress + INT8 dynamic quant + insertslice opt
```shell
python3 test_ov.py -m ../../../MiniCPM3-4B -ov MiniCPM3-4B-ov -llm_int4_com -llm_int8_quant                    
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
OpenVINO version 
 2025.0.0-17426-287ab9883ac


你是谁
你好，我是MiniCPM系列模型，由面壁智能和OpenBMB开源社区开发。详细信息可以在我的简介中找到。


LLM Model First token latency: 86.21 ms, Output len: 28, Avage token latency: 55.34 ms
```
### Test with LNL IGPU
```shell
python test_ov.py -m MiniCPM3-4B-ov -ov MiniCPM3-4B-ov -d GPU -llm_int4_com -llm_int8_quant
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
OpenVINO version
 2024.6.0-17404-4c0f47d2335-releases/2024/6


你是谁
你好，我是MiniCPM系列模型，由面壁智能和OpenBMB开源社区开发。详细信息可以在我的简介中找到。


LLM Model First token latency: 183.88 ms, Output len: 28, Avage token latency: 44.60 ms
```
### Note:
After the command is executed, the IR of OpenVINO will be saved in the directory /path/to/MiniCPM3-4B-ov. If the existence of /path/to/MiniCPM3-4B-ov is detected, the model conversion process will be skipped and the IR of OpenVINO will be loaded directly.

If you only want to convert the model, you can add the -convert_model_only parameter:
```shell
python3 test_ov.py -m /path/to/MiniCPM3-4B -ov MiniCPM3-4B-ov -convert_model_only
```

The model: [Model link](https://hf-mirror.com/openbmb/MiniCPM3-4B)
### Parsing test_ov.py's arguments :
```shell
usage: Export MiniCPM3-4B Model to IR [-h] [-m MODEL_ID] -ov OV_IR_DIR [-d DEVICE] [-p PROMPT] [-max MAX_NEW_TOKENS] [-llm_int4_com] [-llm_int8_quant] [-convert_model_only]

options:
  -h, --help            show this help message and exit
  -m MODEL_ID, --model_id MODEL_ID
                        model_id or directory for loading
  -ov OV_IR_DIR, --ov_ir_dir OV_IR_DIR
                        output directory for saving model
  -d DEVICE, --device DEVICE
                        inference device
  -p PROMPT, --prompt PROMPT
                        prompt
  -max MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        max_new_tokens
  -llm_int4_com, --llm_int4_compress
                        llm int4 weights compress
  -llm_int8_quant, --llm_int8_quant
                        llm int8 weights quantize
  -convert_model_only, --convert_model_only
                        convert model to ov only, do not do inference test
```

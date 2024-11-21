from minicpmv3_helper import convert_minicpmv3_model
from pathlib import Path
import nncf
model_id = "MiniCPM3-4B"
out_dir = "MiniCPM3-4B-ov"
compression_configuration = {
    "mode": nncf.CompressWeightsMode.INT4_SYM,
    "group_size": 128,
    "ratio": 1.0,
}
compression_configuration=None
convert_minicpmv3_model(model_id, out_dir, compression_configuration)
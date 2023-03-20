# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import subprocess
import os

def download_model():
    original_model_name = "openai/whisper-large-v2"
    compute_type = "int8_float16"
    model_path = "whisper-large-v2-ct2"
    subprocess.run(f"ct2-transformers-converter --model {original_model_name} --output_dir {model_path} --copy_files tokenizer.json --quantization {compute_type}", shell=True)

if __name__ == "__main__":
    download_model()
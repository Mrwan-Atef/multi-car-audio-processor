import os
from faster_whisper import WhisperModel

# Define your custom directory
model_path = r"D:\multi-car-audio-processor\models\fast-whisper"

# Ensure the directory exists
os.makedirs(model_path, exist_ok=True)

# Initialize the model and point it to your custom folder
model = WhisperModel(
    "large-v3", 
    device="cpu",               # Change to "cuda" if using an NVIDIA GPU
    compute_type="int8",        # Change to "float16" if using a GPU
    download_root=model_path
)

print(f"Model successfully saved to: {model_path}")

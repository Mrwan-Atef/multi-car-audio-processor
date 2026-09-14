import os
from faster_whisper import WhisperModel

# 1. Configuration Paths
model_path = r"D:\multi-car-audio-processor\models\fast-whisper"

# !!! REPLACE THIS WITH THE ACTUAL PATH TO YOUR AUDIO FILE !!!
your_audio_file = r"D:\multi-car-audio-processor\tests\sample4.m4a" 

# 2. Safety check to make sure your audio file exists
if not os.path.exists(your_audio_file):
    raise FileNotFoundError(f"Could not find your audio file at: {your_audio_file}\n"
                            f"Please check the file path and try again.")

# 3. Load the model from your D: drive folder
print("\nLoading Faster-Whisper large-v3...")
model = WhisperModel(
    "large-v3", 
    device="cpu",               # Change to "cuda" if you have an NVIDIA GPU setup
    compute_type="int8",        # Change to "float16" if using a GPU
    download_root=model_path
)

# 4. Transcribe your local file
print(f"\nTranscribing: {os.path.basename(your_audio_file)}...")
segments, info = model.transcribe(your_audio_file, beam_size=5)

print(f"Detected language: '{info.language}' (Confidence: {info.language_probability * 100:.1f}%)")

# 5. Output the results
print("\n--- Transcription ---")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
print("---------------------")

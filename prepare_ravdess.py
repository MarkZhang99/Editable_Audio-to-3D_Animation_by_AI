import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import random

# Directory paths
AUDIO_DIR = "/transfer/emotalk/EmoTalk_release-main/RAVDESS/audio/"  # .wav files
BLENDSHAPE_DIR = "/transfer/emotalk/EmoTalk_release-main/RAVDESS/blendshape/"  # .npy files
OUTPUT_TRAIN_DIR = "/transfer/emotalk/EmoTalk_release-main/data/RAVDESS/train/"
OUTPUT_VAL_DIR = "/transfer/emotalk/EmoTalk_release-main/data/RAVDESS/val/"
OUTPUT_TEST_DIR = "/transfer/emotalk/EmoTalk_release-main/data/RAVDESS/test/"
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_VAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

# Emotion label map (from filename)
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

emotion_to_index = {emo: i for i, emo in enumerate(emotion_map.values())}

# Filter emotions (optional)
selected_emotions = {"happy", "sad", "angry", "fearful", "disgust", "surprised"}  # skip neutral/calm

# Collect valid file stems
valid_files = []
for fname in os.listdir(BLENDSHAPE_DIR):
    if not fname.endswith(".npy"):
        continue
    parts = fname.replace(".npy", "").split("-")
    if len(parts) != 7:
        continue
    emotion_code = parts[2]
    emotion = emotion_map.get(emotion_code, None)
    if emotion is None or emotion not in selected_emotions:
        continue
    base_name = fname.replace(".npy", "")
    audio_path = os.path.join(AUDIO_DIR, base_name + ".wav")
    blendshape_path = os.path.join(BLENDSHAPE_DIR, fname)
    if os.path.exists(audio_path):
        valid_files.append(fname)

# Shuffle & split (80/10/10)
random.shuffle(valid_files)
n = len(valid_files)
nt = int(n*0.8)
vn = int(n*0.1)
train_list = valid_files[:nt]
val_list   = valid_files[nt:nt+vn]
test_list  = valid_files[nt+vn:]
print(f"Total={n}, train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

# Helper to save

def save_split(file_list, out_dir):
    for fname in tqdm(file_list, desc=f"Saving to {out_dir}"):
        parts = fname.replace(".npy", "").split("-")
        emotion_code = parts[2]
        emotion = emotion_map.get(emotion_code, None)
        actor_id = int(parts[6])
        base_name = fname.replace(".npy", "")

        audio_path = os.path.join(AUDIO_DIR, base_name + ".wav")
        blendshape_path = os.path.join(BLENDSHAPE_DIR, fname)

        # Load data
        audio_waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio_waveform = torchaudio.functional.resample(audio_waveform, sr, 16000)

        blendshape = np.load(blendshape_path)
        blendshape = torch.tensor(blendshape).float()  # (F, 52)

        # Save .pt
        out = {
            'audio': audio_waveform.squeeze(0),
            'blendshape': blendshape,
            'level': emotion_to_index[emotion],
            'person': actor_id
        }
        torch.save(out, os.path.join(out_dir, base_name + ".pt"))

# Process both splits
save_split(train_list, OUTPUT_TRAIN_DIR)
save_split(val_list, OUTPUT_VAL_DIR)
save_split(test_list, OUTPUT_TEST_DIR)
print("✅ split into train/val/test complete.")

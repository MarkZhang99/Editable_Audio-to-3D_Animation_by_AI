import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import random

# Directory paths
AUDIO_DIR = "/transfer/emotalk/EmoTalk_release-main/HDTF/audio"
BLENDSHAPE_DIR = "/transfer/emotalk/EmoTalk_release-main/HDTF/blendshape"
OUTPUT_TRAIN_DIR = "/transfer/emotalk/EmoTalk_release-main/data/HDTF/train"
OUTPUT_VAL_DIR = "/transfer/emotalk/EmoTalk_release-main/data/HDTF/val"
OUTPUT_TEST_DIR = "/transfer/emotalk/EmoTalk_release-main/data/HDTF/test"

os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_VAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)
# Collect file stems (without extension)
file_stems = [f.replace(".wav", "") for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
random.shuffle(file_stems)
split_idx = len(file_stems)
# Train/val split
nt = int(split_idx*0.8)
vn = int(split_idx*0.1)
train_stems = file_stems[:nt]
val_stems = file_stems[nt:nt+vn]
test_stems  = file_stems[nt+vn:]
print(f"Total={split_idx}, train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")




# Helper function to get person ID from file name
def extract_person_id(name):
    parts = name.split('_')
    if len(parts) >= 2:
        return hash(parts[1]) % 1000  # use speaker name for person ID
    return 0

# Process one split

def process_split(split_stems, out_dir):
    for stem in tqdm(split_stems, desc=f"Processing {out_dir}"):
        audio_path = os.path.join(AUDIO_DIR, stem + ".wav")
        blendshape_path = os.path.join(BLENDSHAPE_DIR, stem + ".npy")

        if not os.path.exists(audio_path) or not os.path.exists(blendshape_path):
            print(f"⚠️ Missing file: {stem}")
            continue

        # Load audio
        audio_waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio_waveform = torchaudio.functional.resample(audio_waveform, sr, 16000)

        # Load blendshape
        blendshape = np.load(blendshape_path)
        blendshape = torch.tensor(blendshape).float()  # (F, 52)

        # Level is set to neutral (0), no emotion in HDTF
        out = {
            'audio': audio_waveform.squeeze(0),
            'blendshape': blendshape,
            'level': 0,
            'person': extract_person_id(stem)
        }

        torch.save(out, os.path.join(out_dir, stem + ".pt"))

# Run both splits
process_split(train_stems, OUTPUT_TRAIN_DIR)
process_split(val_stems, OUTPUT_VAL_DIR)
process_split(test_stems, OUTPUT_TEST_DIR)
print("✅ split into train/val/test complete.")

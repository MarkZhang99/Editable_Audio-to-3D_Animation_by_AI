# EmoTalk_v0002: Enhanced Speech-Driven Emotional 3D Face Animation

> ⚠ This project is based on the original **EmoTalk** (ICCV 2023) by Ziqiao Peng et al. Please refer to the [original repository](https://github.com/psyai-net/EmoTalk_release) for baseline implementation and citation. This fork introduces architectural enhancements and fine-tuning pipelines to improve emotional expressiveness, motion precision, and controllability.

## ✨ What's New in EmoTalk_v0002?

EmoTalk_v0002 builds upon the original EmoTalk by introducing several key upgrades:

- ✅ **FiLM-based emotion modulation**: enables learnable fusion between emotional and phonetic features.
- ✅ **GRU-based temporal emotion encoding**: captures dynamic emotional variations across time.
- ✅ **Region-weighted blendshape loss**: enhances facial articulation accuracy, especially in lips and brows.
- ✅ **Level and identity control embeddings**: supports controllable emotional intensity and speaker style.
- ✅ **Editable blendshape output**: outputs 52D FLAME-compatible blendshape sequences for real-time animation in Blender.

## 🧠 Research Context

This project was conducted as an academic extension to explore:
- Emotional controllability in speech-driven animation;
- Editable facial animation pipelines compatible with Blender and FLAME;
- Trade-offs in fine-tuning with emotion-less datasets (e.g., HDTF).

## 📦 Installation

### Environment
```bash
conda create -n emotalk_v0002 python=3.8
conda activate emotalk_v0002
pip install -r requirements.txt

### Blender Setup
wget https://ftp.nluug.nl/pub/graphics/blender/release/Blender3.4/blender-3.4.1-linux-x64.tar.xz
tar -xf blender-3.4.1-linux-x64.tar.xz
mv blender-3.4.1-linux-x64 blender

## 🚀 Demo

Download my pretrained checkpoint (trained on RAVDESS):

📦 `emotalk_finetuned.pth`  
🔗 [Download link](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/s5727214_bournemouth_ac_uk/EkeguJzWxFdLhgSd0LMsWuYB36y_0sAJhMnkWQcs0DA0zg?e=XXN140)

Place the file under the project root directory.

Then run:

```bash
python demo_finetuned.py \
  --wav_path "./audio/online_audio/alive.wav" \
  --result_path "./result/exp_result/7" \
  --model_path "./emotalk_finetuned.pth" \
  --level 7 \
  --person 1

python demo_finetuned.py --wav_path "./audio/online_audio/alive.wav" --result_path "./result/exp_result/7" --model_path "./emotalk_finetuned.pth" --level 7 --person 1

## Datasets
This project uses the same datasets as the original EmoTalk work (from the 3DETF dataset):
RAVDESS — Emotion-annotated speech audio.
HDTF — High-resolution talking face videos without emotion labels.
The dataset preprocessing scripts are:
prepare_ravdess.py
prepare_hdtf.py

##Acknowledgements
This project builds upon the excellent work of the original EmoTalk authors. We sincerely thank them for releasing their code and dataset.

Credits and dependencies include:
EmoTalk (ICCV 2023) – Base model and design.
3DETF dataset – Includes RAVDESS and HDTF data.
Wav2Vec2 pretrained models – For speech representation extraction.
FLAME model & Blender integration – For mesh reconstruction and visualization.

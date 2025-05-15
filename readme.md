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

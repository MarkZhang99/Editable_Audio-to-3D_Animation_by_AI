import os
import torch
from Emotalk_model_v0003 import EmoTalk  # 确保这是修改过 alias 的新版
from types import SimpleNamespace

# —— 1. 和旧模型用的一样的 args —— #
args_v2 = SimpleNamespace(
    transformer_dim    = 512,
    transformer_heads  = 4,
    transformer_layers = 1,
    emo_gru_hidden     = 128,
    emo_gru_layers     = 2,
    num_emotions       = 2,
    num_person         = 24,
    max_seq_len        = 512,
    period             = 20,
    bs_dim             = 52,
    device             = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size = 1
)

DEVICE = args_v2.device
OLD_CKPT = '/transfer/emotalk/EmoTalk_release-main/pretrain_model/EmoTalk.pth'
NEW_CKPT = 'aligned_v2_init.pth'

# —— 2. 加载旧权重字典 —— #
old_state = torch.load(OLD_CKPT, map_location=DEVICE)

# —— 3. 定义 key 前缀映射 —— #
prefix_map = {
    'audio_encoder_cont.':    'extractor.',       # Wav2Vec2Model
    'processor.':             'feature_extractor.',# feature extractor
    'audio_feature_map_cont.': 'extractor_proj.', # 内容投射
    'audio_feature_map_emo.':   'audio_feature_map_emo.', # 情感投射（名字相同）
    'audio_feature_map_emo2.':  'audio_feature_map_emo2.',# 二级情感投射
    'transformer_decoder.':    'decoder.',         # TransformerDecoder
    'bs_map_r.':               'linear.'           # 输出线性层
}

# —— 4. 重写旧 state_dict key —— #
remapped_state = {}
for k, v in old_state.items():
    new_k = k
    for old_pf, new_pf in prefix_map.items():
        if k.startswith(old_pf):
            new_k = new_pf + k[len(old_pf):]
            break
    remapped_state[new_k] = v

# —— 5. 实例化新版并拿到它的 state_dict —— #
model_v2 = EmoTalk(args_v2).to(DEVICE)
new_state = model_v2.state_dict()

# —— 6. 筛选 shape 匹配的 tensor —— #
matched = {}
for k, v in remapped_state.items():
    if k in new_state and v.size() == new_state[k].size():
        matched[k] = v

print(f"🔑 Matched keys:   {len(matched)}")
print(f"❌ Old-only keys:  {len(old_state)-len(matched)}")
print(f"❌ New-only keys:  {len(new_state)-len(matched)}")

# —— 7. 合并 & 加载 —— #
new_state.update(matched)
info = model_v2.load_state_dict(new_state, strict=False)
print("➡️  Loaded with strict=False")
print("   missing_keys:", info.missing_keys)
print("   unexpected_keys:", info.unexpected_keys)

# —— 8. 保存新的 warm-start 模型 —— #
torch.save(model_v2.state_dict(), NEW_CKPT)
print(f"✅ Saved aligned init model to {NEW_CKPT}")

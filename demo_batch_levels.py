import os
import argparse
import torch
import librosa
import numpy as np
from emotalk_model_v0002_upgrade import EmoTalkUpgrade
from scipy.signal import savgol_filter
from demo_upgrade import render_video, get_audio_duration_sec

@torch.no_grad()
def run_one_level(args, level):
    model = EmoTalk(args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device), strict=False)
    model.to(args.device)
    model.eval()

    # 加载音频
    speech_array, sr = librosa.load(args.wav_path, sr=16000)
    audio = torch.FloatTensor(speech_array).unsqueeze(0).to(args.device)

    level_tensor = torch.tensor([level]).to(args.device)
    person_tensor = torch.tensor([args.person]).to(args.device)

    prediction = model.predict(audio, level_tensor, person_tensor)
    prediction = prediction.squeeze(0).cpu().numpy()

    if args.post_processing:
        output = np.zeros_like(prediction)
        for i in range(prediction.shape[1]):
            output[:, i] = savgol_filter(prediction[:, i], 5, 2)
        prediction = output

    # 保存
    # 保存 prediction
    base_name = os.path.splitext(os.path.basename(args.wav_path))[0]
    output_name = f"{base_name}_emo{level}"
    save_path = os.path.join(args.result_path, f"{output_name}.npy")
    np.save(save_path, prediction)

    # 渲染：把 wav_name 改成当前保存的 output_name
    class DummyArgs:
        wav_path = args.wav_path
        result_path = args.result_path
        blender_path = args.blender_path
        wav_name = output_name  # 👈加这个！

    print(f"🎞️ 渲染 {output_name} 中...")
    render_video(DummyArgs(), frame_count=prediction.shape[0])


def main():
    p = argparse.ArgumentParser()
    # —— demo_upgrade-specific args —— 
    p.add_argument("--wav_path",     type=str, default="./input.wav")
    p.add_argument("--model_path",   type=str, default="./emotalk_upgrade_finetuned.pth")
    p.add_argument("--output_dir",   type=str, default="./result")
    p.add_argument("--wav_name",     type=str, default="demo")
    p.add_argument("--blender_path", type=str, default="./blender/blender")
    p.add_argument("--render_blend", type=str, default="./render.blend")
    p.add_argument("--render_py",    type=str, default="./render.py")
    p.add_argument("--post_processing", action="store_true", default=True)
    p.add_argument("--smooth_window", type=int, default=5)

    # —— model hyperparams (must match v0002_upgrade) —— 
    p.add_argument("--bs_dim",            type=int,   default=52)
    p.add_argument("--max_seq_len",       type=int,   default=512)
    p.add_argument("--period",            type=int,   default=20)
    p.add_argument("--emo_gru_hidden",    type=int,   default=128)
    p.add_argument("--emo_gru_layers",    type=int,   default=2)
    p.add_argument("--transformer_layers",type=int,   default=4)
    p.add_argument("--transformer_heads", type=int,   default=8)
    p.add_argument("--transformer_dim",   type=int,   default=512)
    p.add_argument("--num_emotions",      type=int,   default=8)
    p.add_argument("--num_person",        type=int,   default=25)
    p.add_argument("--dropout",           type=float, default=0.1)
    p.add_argument("--use_prosody",       action="store_true")
    p.add_argument("--device",            type=str,   default="cuda")

    args = p.parse_args()

    for level in range(args.num_emotions):
        print(f"\n🧪 正在测试情绪等级 {level} ...")
        run_one_level(args, level)

if __name__ == "__main__":
    main()

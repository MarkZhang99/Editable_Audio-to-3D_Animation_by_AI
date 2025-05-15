import librosa
import numpy as np
import argparse
from scipy.signal import savgol_filter
import torch
from Emotalk_model_v0003 import EmoTalk
import random
import os, subprocess, shlex
import wave, contextlib

# ==== 定义眨眼形状 (7 帧) ====
eye1 = np.array([0.00, 0.25, 0.50, 1.00, 0.50, 0.25, 0.00])
eye2 = np.array([0.00, 0.30, 0.60, 1.00, 0.60, 0.30, 0.00])
eye3 = np.array([0.00, 0.20, 0.40, 1.00, 0.40, 0.20, 0.00])
eye4 = np.array([0.00, 0.15, 0.35, 1.00, 0.35, 0.15, 0.00])

def get_audio_duration_sec(wav_path):
    with contextlib.closing(wave.open(wav_path,'r')) as f:
        frames = f.getnframes()
        rate   = f.getframerate()
        return frames / float(rate)

def render_video(args, frame_count=None):
    wn = os.path.basename(args.wav_path).split('.')[0]
    image_dir   = os.path.join(args.result_path, wn)
    os.makedirs(image_dir, exist_ok=True)
    image_temp  = os.path.join(image_dir, "%d.png")
    output_path = os.path.join(args.result_path, wn + ".mp4")

    # 调用 Blender
    cmd = f'{args.blender_path} -t 64 -b render.blend -P render.py -- "{args.result_path}" "{wn}"'
    subprocess.run(shlex.split(cmd), check=True)
    # 合成视频
    if frame_count is not None:
        duration = get_audio_duration_sec(args.wav_path)
        fps = frame_count / duration
    else:
        fps = 30
    ff = f'ffmpeg -r {fps:.2f} -i "{image_temp}" -i "{args.wav_path}" -pix_fmt yuv420p -s 512x768 "{output_path}" -y'
    subprocess.run(ff, shell=True)
    subprocess.run(f'rm -rf "{image_dir}"', shell=True)
    print(f"✅ Generated video: {output_path}")

@torch.no_grad()
def test(args):
    # 1) 加载音频
    y, sr = librosa.load(args.wav_path, sr=16000)
    audio = torch.from_numpy(y).unsqueeze(0).to(args.device)  # [1, L]

    # 2) 构造 level/person 张量（batch-size=1）
    lvl = torch.tensor([args.level], dtype=torch.long, device=args.device)
    prs = torch.tensor([args.person], dtype=torch.long, device=args.device)

    # 3) 模型前向
    model = EmoTalk(args).to(args.device)
    ckpt  = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    pred = model(audio, lvl, prs)          # [1, T, bs_dim]
    pred = pred.squeeze(0).cpu().numpy()   # [T, bs_dim]

    # 4) 保存 .npy（Blender expect: result_path/wav_name.npy）
    wn = os.path.basename(args.wav_path).split('.')[0]
    save_path = os.path.join(args.result_path, f"{wn}.npy")
    os.makedirs(args.result_path, exist_ok=True)

    if args.post_processing:
        bs = savgol_filter(pred, args.smooth_window, 2, axis=0)
        bs[:, 8] = bs[:, 9] = 0
        i = random.randint(0, 60)
        while i < bs.shape[0] - 7:
            shape = random.choice([eye1, eye2, eye3, eye4])
            bs[i:i+7, 8] = bs[i:i+7, 9] = shape
            i += random.randint(60, 180)
        np.save(save_path, bs)
        print(f"💾 Saved post-processed predictions to {save_path}")
        return bs
    else:
        np.save(save_path, pred)
        print(f"💾 Saved raw predictions to {save_path}")
        return pred

def main():
    parser = argparse.ArgumentParser(description="Demo EmoTalk v0003")
    # I/O
    parser.add_argument("--wav_path",     type=str, default="./input.wav")
    parser.add_argument("--model_path",   type=str, default="phase2_ckpt.pth")
    parser.add_argument("--result_path",  type=str, default="./output",
                        help="Where {wav_name}.npy and .mp4 will be saved")
    parser.add_argument("--blender_path", type=str, default="./blender/blender")

    # Model hyperparams (must match v0003)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--transformer_heads",  type=int, default=4)
    parser.add_argument("--transformer_dim",    type=int, default=512)
    parser.add_argument("--bs_dim",             type=int, default=52)
    parser.add_argument("--max_seq_len",        type=int, default=512)
    parser.add_argument("--period",             type=int, default=20)
    parser.add_argument("--emo_gru_hidden",     type=int, default=128)
    parser.add_argument("--emo_gru_layers",     type=int, default=2)
    parser.add_argument("--num_emotions",       type=int, default=8)
    parser.add_argument("--num_person",         type=int, default=24)

    # Inference params
    parser.add_argument("--level",          type=int, default=1,
                        help="Emotion level (1..num_emotions)")
    parser.add_argument("--person",         type=int, default=1,
                        help="Person ID (1..num_person)")
    parser.add_argument("--smooth_window",  type=int, default=5,
                        help="Savgol filter window length (odd int)")
    parser.add_argument("--post_processing",action="store_true", default=True,
                        help="Apply smoothing + blinking post-processing")

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred = test(args)
    frame_count = pred.shape[0]
    print(f"🧠 Generated {frame_count} frames of blendshapes.")
    render_video(args, frame_count=frame_count)

if __name__ == "__main__":
    main()

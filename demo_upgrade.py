# demo_upgrade.py
import os, argparse, shlex, subprocess, wave, contextlib, random
import torch, librosa, numpy as np
from scipy.signal import savgol_filter
from emotalk_model_v0002_upgrade import EmoTalkUpgrade
from utils import init_biased_mask, enc_dec_mask

# ==== helpers ====
def get_audio_duration_sec(wav_path):
    with contextlib.closing(wave.open(wav_path,'r')) as f:
        return f.getnframes() / float(f.getframerate())

@torch.no_grad()
def test(args):
    # —— 填充 Upgrade 模型需要的 args 字段 —— 
    args.bs_dim            = args.bs_dim
    args.device            = torch.device(args.device)
    args.max_seq_len       = args.max_seq_len
    args.period            = args.period
    args.emo_gru_hidden    = args.emo_gru_hidden
    args.emo_gru_layers    = args.emo_gru_layers
    args.transformer_layers= args.transformer_layers
    args.transformer_heads = args.transformer_heads
    args.transformer_dim   = args.transformer_dim
    args.num_emotions      = args.num_emotions
    args.num_person        = args.num_person
    args.dropout           = args.dropout
    args.use_prosody       = args.use_prosody

    os.makedirs(args.output_dir, exist_ok=True)

    # —— 载入眨眼模板 —— 
    eye1 = np.array([0.36537236,0.950235724,0.95593375,0.916715622,
                     0.367256105,0.119113259,0.025357503])
    eye2 = np.array([0.234776169,0.909951985,0.944758058,0.777862132,
                     0.191071674,0.235437036,0.089163929])
    eye3 = np.array([0.870040774,0.949833691,0.949418545,0.695911646,
                     0.191071674,0.072576277,0.007108896])
    eye4 = np.array([0.000307991,0.556701422,0.952656746,0.942345619,
                     0.425857186,0.148335218,0.017659493])

    # —— load model —— 
    model = EmoTalkUpgrade(args).to(args.device)
    ckpt = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # —— load audio —— 
    y, sr = librosa.load(args.wav_path, sr=16000)
    audio = torch.from_numpy(y).unsqueeze(0).to(args.device)
    level  = torch.tensor([args.level], dtype=torch.long, device=args.device)
    person = torch.tensor([args.person], dtype=torch.long, device=args.device)

    # —— forward & postproc —— 
    bs_out, _ = model.predict(audio, level, person), None
    pred = bs_out.squeeze(0).cpu().numpy()
    # —— ： smoothing + blink —— 
    if args.post_processing:
        # Savitzky–Golay 平滑
        win = args.smooth_window if args.smooth_window %2==1 else args.smooth_window+1
        for i in range(pred.shape[1]):
            pred[:,i] = savgol_filter(pred[:,i], win, 2)

    
        pred[:,8] = 0
        pred[:,9] = 0

        i = random.randint(0, 60)
        while i < pred.shape[0] - 7:
            eye_choice = random.choice([eye1, eye2, eye3, eye4])
            pred[i:i+7, 8] = eye_choice
            pred[i:i+7, 9] = eye_choice
            i += random.randint(60, 180)

    # —— save —— 
    np.save(f"{args.output_dir}/{args.wav_name}.npy", pred)
    torch.cuda.empty_cache()
    return pred


def render_video(args, frame_count):
    image_dir  = f"{args.output_dir}/{args.wav_name}"
    os.makedirs(image_dir, exist_ok=True)
    blender_cmd = shlex.split(
        f'{args.blender_path} -t 64 -b {args.render_blend} -P {args.render_py} -- '
        f'"{args.output_dir}" "{args.wav_name}"'
    )
    subprocess.run(blender_cmd, check=True)
    fps = frame_count / get_audio_duration_sec(args.wav_path)
    mp4 = f"{args.output_dir}/{args.wav_name}.mp4"
    subprocess.run(
        f'ffmpeg -r {fps:.2f} -i "{image_dir}/%d.png" -i "{args.wav_path}" '
        f'-pix_fmt yuv420p -s 512x768 "{mp4}" -y', shell=True
    )
    print(f"✅ Video saved to {mp4}")

if __name__ == "__main__":
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

    # —— session args —— 
    p.add_argument("--level",  type=int, default=1)
    p.add_argument("--person", type=int, default=1)
    args = p.parse_args()

    pred = test(args)
    render_video(args, frame_count=pred.shape[0])

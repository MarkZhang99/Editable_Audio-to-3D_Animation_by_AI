# upgrade_train.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from emotalk_model_v0002_upgrade import EmoTalkUpgrade
from train import DummyDataset
from loss_v0002 import EmoTalkLoss
from types import SimpleNamespace
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args():
    p = argparse.ArgumentParser("Train EmoTalkUpgrade")
    # —— Training HP —— #
    p.add_argument("--epochs",     type=int,   default=10,     help="epochs")
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--lr",         type=float, default=5e-6,   help="lr")
    p.add_argument("--cls_alpha",  type=float, default=0.1,    help="cls")
    p.add_argument("--patience",   type=int,   default=3,      help="EarlyStopping patience")
    p.add_argument("--save_path",  type=str,   default="best_ckpt.pth", help="save path")
    p.add_argument("--test_id",    type=str,   default="",     help="test ID")
    # —— model HP —— #
    p.add_argument("--bs_dim",             type=int,   default=52)
    p.add_argument("--max_seq_len",        type=int,   default=512)
    p.add_argument("--period",             type=int,   default=20)
    p.add_argument("--emo_gru_hidden",     type=int,   default=128)
    p.add_argument("--emo_gru_layers",     type=int,   default=2)
    p.add_argument("--transformer_layers", type=int,   default=4)
    p.add_argument("--transformer_heads",  type=int,   default=8)
    p.add_argument("--transformer_dim",    type=int,   default=512)
    p.add_argument("--num_emotions",       type=int,   default=8)
    p.add_argument("--num_person",         type=int,   default=25)
    p.add_argument("--dropout",            type=float, default=0.1)
    p.add_argument("--use_prosody",        action="store_true", help="如果有 pitch/energy 特征就打开")
    return p.parse_args()

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # —— DataLoaders —— #
    train_loader = DataLoader(DummyDataset('./data/RAVDESS/train'),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(DummyDataset('./data/RAVDESS/val'),
                              batch_size=args.batch_size, shuffle=False)

    # —— Model / Loss / Optimizer —— #
    model   = EmoTalkUpgrade(args).to(device)
    loss_fn = EmoTalkLoss(region_weighted=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    best_val_bs = float('inf')
    no_improve  = 0

    for epoch in range(1, args.epochs + 1):
        # ——— Training ——— #
        model.train()
        running_bs  = 0.0
        running_cls = 0.0
        logs_accum  = {"main":0, "smooth":0, "vel":0}

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{args.epochs}"):
            audio      = batch["audio"].to(device)
            blendshape = batch["blendshape"].to(device)
            level      = batch["level"].to(device)
            person     = batch["person"].to(device)
            args.target_len = blendshape.size(1)

            # forward
            bs_pred, cls_logit = model(
                audio, level, person,
                prosody=None if not args.use_prosody else batch["prosody"].to(device)
            )
            bs_loss, logs = loss_fn(bs_pred, blendshape)
            cls_loss      = F.cross_entropy(cls_logit, level)
            loss          = bs_loss + args.cls_alpha * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_bs  += bs_loss.item()
            running_cls += cls_loss.item()
            for k, v in logs.items():
                if k in logs_accum:
                    logs_accum[k] += v

        avg_trn_bs  = running_bs  / len(train_loader)
        avg_trn_cls = running_cls / len(train_loader)
        train_losses.append(avg_trn_bs + args.cls_alpha * avg_trn_cls)
        print(f"→ Train BS {avg_trn_bs:.4f} | CLS {avg_trn_cls:.4f} | "
              f"main:{logs_accum['main']:.1f} smooth:{logs_accum['smooth']:.1f} vel:{logs_accum['vel']:.1f}")

        # ——— Validation ——— #
        model.eval()
        running_val_bs = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[ Val  ] Epoch {epoch}/{args.epochs}"):
                audio      = batch["audio"].to(device)
                blendshape = batch["blendshape"].to(device)
                level      = batch["level"].to(device)
                person     = batch["person"].to(device)
                args.target_len = blendshape.size(1)

                bs_pred, _ = model(
                    audio, level, person,
                    prosody=None if not args.use_prosody else batch["prosody"].to(device)
                )
                val_bs, _  = loss_fn(bs_pred, blendshape)
                running_val_bs += val_bs.item()

        avg_val_bs = running_val_bs / len(val_loader)
        val_losses.append(avg_val_bs)
        print(f"→ Val   BS {avg_val_bs:.4f}")

        # —— Early Stopping & Best Model —— #
        if avg_val_bs < best_val_bs:
            best_val_bs = avg_val_bs
            no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved at epoch {epoch} | Val BS {best_val_bs:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"\n Training finished. Best Val BS Loss: {best_val_bs:.4f}")
    print("Best model saved to", args.save_path)

    # —— Plot Loss Curves —— #
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train (BS+CLS)")
    plt.plot(val_losses,   label="Val   (BS)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # name with test_id 
    out_png = (args.test_id + "_loss_curve.png") if args.test_id else "loss_curve.png"
    plt.savefig(out_png)
    plt.show()

if __name__ == "__main__":
    main()

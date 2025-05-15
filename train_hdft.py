import torch
from torch.utils.data import DataLoader
from Emotalk_model_v0002 import EmoTalk
from loss_v0003 import EmoTalkLoss, pseudo_accuracy
import os
import numpy as np
from tqdm import tqdm
import random
# ======= 示例数据集结构 ========
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.data_paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith('.pt')
        ]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data = torch.load(self.data_paths[index])
        
        # ✅ DEBUG 输出
        level = data['level']
        person = data['person']
        if not (0 <= level <= 8):
            print(f"⚠️ level 超出范围：{level} at index {index}（路径：{self.data_paths[index]}）")
            level = min(max(level, 0), 8)  # clip 到合法范围
        if not (0 <= person < 25):
            print(f"⚠️ person 超出范围：{person} at index {index}")
            person = min(max(person, 0), 24)

        return {
            'audio': data['audio'],
            'blendshape': data['blendshape'],
            'level': torch.tensor(level, dtype=torch.long),  # 保证类型
            'person': torch.tensor(person, dtype=torch.long)
        }

class HDTFDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.data_paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith('.pt')
        ]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        #MAX_AUDIO_LEN = 640000
        data = torch.load(self.data_paths[index])
        audio = data['audio']

        # ✅ 如果是 1D，则扩展维度
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # ✅ 随机裁切片段（提升多样性）
        #if audio.shape[1] > MAX_AUDIO_LEN:
            #start = random.randint(0, audio.shape[1] - MAX_AUDIO_LEN)
            #audio = audio[:, start:start + MAX_AUDIO_LEN]

        return {
            'audio': audio,
            'blendshape': data['blendshape'],
            'level': torch.tensor(0, dtype=torch.long),
            'person': torch.tensor(0, dtype=torch.long)
        }

# ======= 训练主函数 ========
def train_model(model, dataloader, optimizer, loss_fn, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for audio, target, level, person in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            audio = audio.to(device)        # (B, T)
            target = target.to(device)      # (B, F, 52)
            level = level.to(device)
            person = person.to(device)

            frame_num = target.shape[1]

            # 特征抽取
            inputs_cont = model.processor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(device)
            cont_feat = model.audio_encoder_cont(inputs_cont, frame_num=frame_num).last_hidden_state
            cont_feat = model.audio_feature_map_cont(cont_feat)

            inputs_emo = model.feature_extractor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
            emo_output = model.audio_encoder_emo(inputs_emo, frame_num=frame_num)
            emo_feat = model.audio_feature_map_emo(emo_output.hidden_states)
            emo_feat_256 = torch.relu(model.audio_feature_map_emo2(emo_feat))

            # 前向传播
            pred = model.forward(cont_feat, emo_feat_256, level, person)

            # 损失计算
            loss, log = loss_fn(pred, target)
            acc = pseudo_accuracy(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += log['total']
            total_acc += acc

        print(f"Epoch {epoch+1}: Loss={total_loss / len(dataloader):.4f} | Acc={total_acc / len(dataloader):.4f}")


# ======= 启动训练 ========
if __name__ == '__main__':
    from argparse import Namespace

    args = Namespace(
        feature_dim=832,
        bs_dim=52,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
        max_seq_len=5000,
        period=30
    )

    model = EmoTalk(args).to(args.device)
    # 选择性 warm start
    if os.path.exists("./pretrain_model/emotalk_finetuned.pth"):
        state_dict = torch.load("./pretrain_model/emotalk_finetuned.pth", map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded pretrained weights")
    
    # ✅ 冻结情绪相关模块
    for name, param in model.named_parameters():
        if 'extract_emotion' in name or 'emo_gru' in name or 'gamma_layer' in name or 'beta_layer' in name:
            param.requires_grad = False
            print(f"❄️  Freezing: {name}")


    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    loss_fn = EmoTalkLoss()
    dataset = DummyDataset("./data")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    train_model(model, dataloader, optimizer, loss_fn, args.device, epochs=10)

    torch.save(model.state_dict(), "./checkpoints/emotalk_v2_final.pth")
    print("✅ Model saved to ./checkpoints/emotalk_v2_final.pth")

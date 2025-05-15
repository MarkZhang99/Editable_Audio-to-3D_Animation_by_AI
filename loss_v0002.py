import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EmoTalk  # 加载结构

# 实际通道名顺序，来自用户定义
model_bsList = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft",
    "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut"
]
def match_length(pred, target):
    """
    自动对齐 pred 和 target 的时间长度
    如果长度不一致，将 target 插值为 pred 的长度
    """
    T_pred, T_gt = pred.size(1), target.size(1)
    if T_pred == T_gt:
        return pred, target
    elif T_pred < T_gt:
        target = F.interpolate(target.transpose(1, 2), size=T_pred, mode='linear').transpose(1, 2)
    else:
        pred = F.interpolate(pred.transpose(1, 2), size=T_gt, mode='linear').transpose(1, 2)
    return pred, target

class RegionWeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        weight_map = []
        for name in model_bsList:
            if any(key in name for key in ['mouth', 'jaw', 'tongue', 'brow' ]):
                weight_map.append(0.4)
            elif any(key in name for key in [ 'eye','cheek', 'nose']):
                weight_map.append(0.3)
            else:
                weight_map.append(0.1)
        self.weight = nn.Parameter(torch.tensor(weight_map).float(), requires_grad=False)

    def forward(self, pred, target):
        return ((pred - target).abs() * self.weight.view(1, 1, -1).to(pred.device)).mean()


def smoothness_loss(pred):
    return ((pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2])**2).mean()


def velocity_loss(pred, target):
    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_gt = target[:, 1:] - target[:, :-1]
    return F.mse_loss(vel_pred, vel_gt)


def pseudo_accuracy(pred, target, threshold=0.05):
    return ((pred - target).abs() < threshold).float().mean().item()


class EmotionClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, emo_embedding, emo_label):
        return self.loss(emo_embedding, emo_label)


class EmoTalkLoss(nn.Module):
    def __init__(self, region_weighted=True):
        super().__init__()
        self.region_loss = RegionWeightedL1Loss() if region_weighted else nn.L1Loss()
        self.smooth_weight = 0.1
        self.vel_weight = 0.5

    def forward(self, pred, target):
        pred, target = match_length(pred, target)

        loss_main = self.region_loss(pred, target)
        loss_smooth = smoothness_loss(pred)
        loss_vel = velocity_loss(pred, target)
        total = loss_main + self.smooth_weight * loss_smooth + self.vel_weight * loss_vel
        return total, {
            "main": loss_main.item(),
            "smooth": loss_smooth.item(),
            "vel": loss_vel.item(),
            "total": total.item()
        }

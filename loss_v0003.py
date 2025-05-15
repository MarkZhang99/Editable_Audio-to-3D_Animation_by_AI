import torch
import torch.nn as nn
import torch.nn.functional as F

# Blendshape 列表（按模型顺序）
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
            if 'mouthSmile' in name or 'jawOpen' in name or 'mouthFunnel' in name:
                weight_map.append(0.5)
            elif any(key in name for key in ['mouth', 'jaw', 'tongue','brow', 'eye']):
                weight_map.append(0.4)
            elif any(key in name for key in [ 'cheek', 'nose']):
                weight_map.append(0.2)
        
            #if 'mouthSmile' in name or 'jawOpen' in name or 'mouthFunnel' in name:
                #weight_map.append(0.5)
            #elif 'mouth' in name or 'jaw' in name or 'tongue' in name:
                #weight_map.append(0.4)
            #elif 'eyeBlink' in name or 'browInnerUp' in name:
                #weight_map.append(0.4)
            #elif any(key in name for key in ['brow', 'eye', 'cheek', 'nose']):
                #weight_map.append(0.3)
            else:
                weight_map.append(0.1)
        self.weight = nn.Parameter(torch.tensor(weight_map).float(), requires_grad=False)

    def forward(self, pred, target):
        pred, target = match_length(pred, target)
        return ((pred - target).abs() * self.weight.view(1, 1, -1).to(pred.device)).mean()

def mouth_focus_loss(pred, target, indices):
    return F.l1_loss(pred[:, :, indices], target[:, :, indices])

def smoothness_loss(pred):
    return ((pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2])**2).mean()

def velocity_loss(pred, target):
    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_gt = target[:, 1:] - target[:, :-1]
    return F.mse_loss(vel_pred, vel_gt)

def pseudo_accuracy(pred, target, threshold=0.05):
    return ((pred - target).abs() < threshold).float().mean().item()

class EmoTalkLoss(nn.Module):
    def __init__(self, region_weighted=True, mouth_only=True, mouth_weight=0.3):
        super().__init__()
        self.region_loss = RegionWeightedL1Loss() if region_weighted else nn.L1Loss()
        self.smooth_weight = 0.4
        self.vel_weight = 0.3
        self.mouth_only = mouth_only
        self.mouth_weight = mouth_weight
        self.mouth_indices = [i for i, name in enumerate(model_bsList) if 'mouth' in name or 'jaw' in name]

    def forward(self, pred, target):
        pred, target = match_length(pred, target)

        loss_main = self.region_loss(pred, target)
        loss_smooth = smoothness_loss(pred)
        loss_vel = velocity_loss(pred, target)

        total = loss_main + self.smooth_weight * loss_smooth + self.vel_weight * loss_vel

        if self.mouth_only:
            loss_mouth = mouth_focus_loss(pred, target, self.mouth_indices)
            total += self.mouth_weight * loss_mouth

        return total, {
            "main": loss_main.item(),
            "smooth": loss_smooth.item(),
            "vel": loss_vel.item(),
            "total": total.item(),
        }

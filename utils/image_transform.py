import torch
import numpy as np
from torchvision import transforms


def to_tensor(image):
    to_tensor = transforms.ToTensor()
    image = to_tensor(np.array(image).astype(np.float32) / 255.0)
    image = torch.clamp(image, 0.0, 1.0)
    return image


def inverse_depth_norm(depth):
    zero_mask = depth == 0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth

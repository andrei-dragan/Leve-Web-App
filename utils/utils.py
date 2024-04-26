import torch


def depth_norm(depth):
    zero_mask = depth == 0
    depth = torch.clamp(depth, 10 / 100, 10)
    depth = 10 / depth
    depth[zero_mask] = 0.0
    return depth


def inverse_depth_norm(depth):
    zero_mask = depth == 0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth

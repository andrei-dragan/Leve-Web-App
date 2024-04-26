import torch.nn as nn
from model.ddrnet import DualResNetBackbone


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.backbone = DualResNetBackbone(pretrained=True, features=64)

    def forward(self, x):
        return self.backbone(x)

import torch.nn as nn
from model.encoder import EncoderLayer
from model.decoder import DecoderLayer


class Leve(nn.Module):
    def __init__(self):
        super(Leve, self).__init__()
        self.encoder = EncoderLayer()
        self.decoder = DecoderLayer(num_features=[64, 32, 16], connection_num_features=[64, 32, 3], connection_upscale_factor=[2, 2, 1])

    def forward(self, x):
        connections, x = self.encoder(x)
        x = self.decoder(x, connections)
        return x

import torch.nn as nn
import torch.nn.functional as F


class LayeredConvolutionBlock(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(LayeredConvolutionBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.bn2 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UpsamplingBlock(nn.Module):

    def __init__(self, num_input_features, num_output_features, connection_num_features, connection_upscale_factor):
        super(UpsamplingBlock, self).__init__()
        self.connection_conv = nn.Conv2d(connection_num_features, num_input_features, kernel_size=1, padding=0)
        self.dual_conv = LayeredConvolutionBlock(num_input_features, num_input_features)
        self.result_conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, padding=0)
        self.connection_upscale_factor = connection_upscale_factor

    def forward(self, x, connection):
        upsampled_x = F.interpolate(x, scale_factor=2, mode="bilinear")

        if self.connection_upscale_factor > 1:
            connection = F.interpolate(connection, scale_factor=self.connection_upscale_factor, mode="bilinear")
        connection = self.connection_conv(connection)

        x = self.dual_conv(upsampled_x)
        x = x + connection + upsampled_x
        x = self.result_conv(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, num_features, connection_num_features, connection_upscale_factor):
        super(DecoderLayer, self).__init__()
        self.upsampling_block1 = UpsamplingBlock(
            num_input_features=num_features[0],
            num_output_features=num_features[1],
            connection_num_features=connection_num_features[0],
            connection_upscale_factor=connection_upscale_factor[0],
        )
        self.upsampling_block2 = UpsamplingBlock(
            num_input_features=num_features[1],
            num_output_features=num_features[2],
            connection_num_features=connection_num_features[1],
            connection_upscale_factor=connection_upscale_factor[1],
        )
        self.upsampling_block3 = UpsamplingBlock(
            num_input_features=num_features[2],
            num_output_features=1,
            connection_num_features=connection_num_features[2],
            connection_upscale_factor=connection_upscale_factor[2],
        )

    def forward(self, x, connections):
        x = self.upsampling_block1(x, connections[2])
        x = self.upsampling_block2(x, connections[1])
        x = self.upsampling_block3(x, connections[0])
        return x

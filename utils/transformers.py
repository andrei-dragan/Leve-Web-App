import torch
import numpy as np
from torchvision import transforms


class ToTensor(object):
    def __init__(self, is_train=True, max_depth=1000.0):
        self.is_train = is_train
        self.max_depth = max_depth

    def __call__(self, image):
        to_tensor = transforms.ToTensor()

        # Transform the image to a tensor and normalize it within the range [0.0, 1.0]
        image = to_tensor(np.array(image).astype(np.float32) / 255.0)
        image = torch.clamp(image, 0.0, 1.0)
        return image

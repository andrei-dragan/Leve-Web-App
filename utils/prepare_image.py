import numpy as np
import random
import os
import time
import math

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from itertools import permutations

from utils.transformers import ToTensor
from utils.utils import depth_norm, inverse_depth_norm


def prepare_image(image):
    image = Image.open(image)

    downscale_image = transforms.Resize((480, 640))
    to_tensor = ToTensor(is_train=False, max_depth=10)

    image = to_tensor(image)
    image = downscale_image(image.unsqueeze(0))
    return image

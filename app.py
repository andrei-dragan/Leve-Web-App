import streamlit as st
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

from utils.infer import infer
from matplotlib import pyplot as plt
import PIL

# create an image input widget
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# print the type of the image
if image is not None:
    output = infer(image)

    # apply viridis colormap
    output = output.squeeze().cpu().numpy()

    # plt.imshow(output, cmap="viridis")
    # plt.axis("off")
    # plt.colorbar()
    # plt.title("Depth map")

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(output)
    st.pyplot(fig)

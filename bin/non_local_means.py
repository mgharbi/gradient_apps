# Hack to avoid launching gtk
import matplotlib 
matplotlib.use('Agg') 

import argparse
import logging
import os
import setproctitle
import time
import gc

import numpy as np
import scipy
import skimage.io
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchlib.viz as viz
import torchlib.utils as utils

import gapps.datasets as datasets
import gapps.modules as models
import gapps.metrics as metrics
import gapps

image = skimage.io.imread("images/rgb.png")
image = image/255.0
image = gapps.utils.make_noisy(image, 0.1)
skimage.io.imsave("output/nlm_input.png", image)
image = image.transpose([2, 0, 1]).astype(np.float32)
image = Variable(th.from_numpy(image), requires_grad=True)

model = models.NonLocalMeans()

output = model(image)

output = output.data.cpu().numpy()
output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
output = np.squeeze(output)

skimage.io.imsave("output/nlm_output.png", output)


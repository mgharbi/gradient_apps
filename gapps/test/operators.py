import os
import time
import unittest
import skimage.io

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler


import gapps.functions.operators as ops

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "data")
out_dir = os.path.join(test_dir, "output")
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

def test_dummy():
  bs = 8;
  c = 3
  h = 16;
  w = 24;
  data = Variable(th.ones(bs, c, h, w))
  output = ops.Dummy.apply(data)

  for c_ in range(c):
    d = output.data[:, c_, ...].numpy()
    assert (np.amax(np.abs(d - c_)) < 1e-5)


def test_bilateral_slice():
  bs = 1;
  gh = 16
  gw = 16
  gd = 8
  c = 3

  guide = skimage.io.imread(os.path.join(data_dir, "gray.png"))
  h, w = guide.shape
  guide = np.expand_dims(guide/255.0, 0).astype(np.float32)

  grid = Variable(th.randn(bs, c, gd, gh, gw))
  guide = Variable(th.from_numpy(guide))
  output = ops.BilateralSlice.apply(grid, guide)

  assert output.shape[0] == bs
  assert output.shape[1] == c
  assert output.shape[2] == h
  assert output.shape[3] == w

  output = np.squeeze(output.data.numpy())
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  skimage.io.imsave(os.path.join(out_dir, "bilateral_slice.png"), output)

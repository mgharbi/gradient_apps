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

import gapps.utils as utils
import gapps.functions.operators as ops

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "data")
out_dir = os.path.join(test_dir, "output")
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

def test_bilateral_layer():
  bs = 1;
  # gh = 16
  # gw = 16
  # gd = 8
  ci = 3
  co = 3
  kh = 3
  kw = 3
  kd = 3

  sx, sy, sz = 8, 8, 8

  # image = skimage.io.imread(os.path.join(data_dir, "rgb.png"))
  # guide = skimage.io.imread(os.path.join(data_dir, "gray.png"))

  image = np.random.uniform(size=(16, 16, 3))
  guide = np.random.uniform(size=(16, 16))

  h, w = guide.shape
  image = np.expand_dims(image.transpose([2, 0 , 1])/255.0, 0).astype(np.float32)
  guide = np.expand_dims(guide/255.0, 0).astype(np.float32)

  image = Variable(th.from_numpy(image), requires_grad=False)
  guide = Variable(th.from_numpy(guide), requires_grad=False)
  kernels = Variable(th.randn(co, ci, kd, kh, kw), requires_grad=True)
  output = ops.BilateralLayer.apply(image, guide, kernels, sx, sy, sz)

  assert output.shape[0] == bs
  assert output.shape[1] == co
  assert output.shape[2] == h - kh
  assert output.shape[3] == w - kw

  loss = output.sum()
  loss.backward()
  # import ipdb; ipdb.set_trace()

  gradcheck(ops.BilateralLayer.apply,
      (image, guide, kernels, sx, sy, sz), eps=1e-4, atol=5e-2, rtol=5e-4,
       raise_exception=True)

  mini, maxi = output.min(), output.max()
  output -= mini
  output /= (maxi-mini)


  output = output.data[0].numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(os.path.join(out_dir, "bilateral_layer.png"), output)


def test_playground():
  bs = 1
  h = 2
  w = 4
  c = 1

  data1 = Variable(th.rand(bs, c, h, w), requires_grad=False)
  data2 = Variable(th.rand(bs, c, h, w), requires_grad=True)
  output = ops.Playground.apply(data1, data2)

  assert output.shape[0] == bs
  assert output.shape[1] == c
  assert output.shape[2] == h
  assert output.shape[3] == w

  # assert ((output.data.numpy()-2) == 0).all()

  loss = output.sum()
  loss.backward()

  gradcheck(ops.Playground.apply,
      (data1, data2), eps=1e-4, atol=5e-2, rtol=5e-4,
       raise_exception=True)

def test_ahd_demosaick():
  image = skimage.io.imread(os.path.join(data_dir, "rgb.png")).astype(np.float32)/255.0
  h, w, _ = image.shape
  mosaick = utils.make_mosaick(image)
  skimage.io.imsave(os.path.join(out_dir, "ahd_mosaick.png"), mosaick)

  mosaick = Variable(th.from_numpy(mosaick), requires_grad=False)
  output = ops.AHDDemosaick.apply(mosaick)

  assert output.shape[0] == 3
  assert output.shape[1] == h
  assert output.shape[2] == w

  output = output.data.numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(os.path.join(out_dir, "ahd_demosaicked.png"), output)

import os
import time
import unittest

# import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler

import gapps.utils as utils
import gapps.functions as funcs
import gapps.modules as modules

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "data")
out_dir = os.path.join(test_dir, "output")
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

def test_learnable_demosaick_cpu():
  _test_learnable_demosaick(False)

def _test_learnable_demosaick(gpu=False):
  # image = skimage.io.imread(
  #     os.path.join(data_dir, "rgb.png")).astype(np.float32)/255.0
  image = np.zeros((128, 128, 3)).astype(np.float32)
  h, w, _ = image.shape
  mosaick = utils.make_mosaick(image)
  # skimage.io.imsave(
  #     os.path.join(out_dir, "learnable_mosaick.png"), mosaick)
  bs = 32
  mosaick = th.from_numpy(mosaick).clone()
  mosaick = mosaick.view(1, 1, h, w)
  mosaick = mosaick.repeat(bs, 1, 1, 1)
  mosaick = Variable(mosaick, requires_grad=True)
  op = modules.LearnableDemosaick()

  if gpu:
    mosaick = mosaick.cuda()
    op.cuda()

  print "starting"
  for i in range(10000):
    print i
    output = op(mosaick)
    # loss = output.sum()
    # loss.backward()
  print "stoping"

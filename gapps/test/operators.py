import os
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler

import gapps.functions.operators as ops

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

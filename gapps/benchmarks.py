import os
import time
import unittest

import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler

import gapps.utils as utils
import gapps.functions as funcs
import gapps.modules as modules


class BenchmarkResult(object):
  def __init__(self, name, time, cuda):
    super(BenchmarkResult, self).__init__()
    self.name = name
    self.time = time
    self.cuda = cuda

  def __repr__(self):
    dev = "gpu" if self.cuda else "cpu"
    s = "{} ({}) {:.2f} ms".format(self.name, dev, self.time)
    return s


class Benchmark(object):
  def __init__(self, cuda=False, burn_iters=5, iters=10):
    super(Benchmark, self).__init__()
    self.cuda = cuda
    self.burn_iters = burn_iters
    self.iters = iters

  def name(self):
    return "EmptyBenchmark"

  def reset(self):
    """Implement this method to set the data and state."""
    pass

  def __call__(self):
    self.reset()
    for i in range(self.burn_iters):
      self.run()

    start = time.time()
    for i in range(self.iters):
      self.run()
    end = time.time()

    runtime = (end-start)*1000.0/self.iters

    return BenchmarkResult(self.name(), runtime, self.cuda)

  def run(self):
    """This method evaluates the critical path of the algortihm to be timed."""
    pass
    
    
class SpatialTransformer(Benchmark):
  def __init__(self, cuda=False, pytorch=False, burn_iters=5, iters=10):
    super(SpatialTransformer, self).__init__(cuda=cuda, burn_iters=burn_iters, iters=iters)
    self.pytorch = pytorch

  def name(self):
    if self.pytorch:
      return "SpatialTransformerPytorch"
    else:
      return "SpatialTransformer"

  def run(self):
    output = self.op(self.image, self.affine_mtx)
    loss = output.sum()
    loss.backward()

  def reset(self):
    sz = 512
    bs = 4
    image = th.randn(bs, 3, sz, sz)

    affine_mtx = th.zeros(bs, 2, 3)
    affine_mtx[:, 0, 1] = 1.0
    affine_mtx[:, 1, 0] = 1.0

    self.image = Variable(image, requires_grad=True)
    self.affine_mtx = Variable(affine_mtx, requires_grad=True)

    self.op = modules.SpatialTransformer(pytorch=self.pytorch)

    if self.cuda:
      self.image = self.image.cuda()
      self.affine_mtx = self.affine_mtx.cuda()
      self.op = self.op.cuda()

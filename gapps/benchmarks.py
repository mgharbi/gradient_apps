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
    with profiler.profile() as prof:
      for i in range(self.iters):
        start1 = time.time()
        self.run()
        # th.cuda.synchronize()
        end1 = time.time()
        runtime1 = (end1-start1)*1000.0
        print "iter {}: {:.2f}ms".format(i, runtime1)
      end = time.time()

    runtime = (end-start)*1000.0/self.iters

    return BenchmarkResult(self.name(), runtime, self.cuda)

  def run(self):
    """This method evaluates the critical path of the algortihm to be timed."""
    pass
    
    
class SpatialTransformer(Benchmark):
  def __init__(self, cuda=False, pytorch=False, burn_iters=5, iters=10):
    super(SpatialTransformer, self).__init__(
        cuda=cuda, burn_iters=burn_iters, iters=iters)
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


class BilateralLayer(Benchmark):
  def __init__(self, cuda=False, pytorch=False, burn_iters=5, iters=10):
    super(BilateralLayer, self).__init__(
        cuda=cuda, burn_iters=burn_iters, iters=iters)
    self.pytorch = pytorch

  def name(self):
    if self.pytorch:
      return "BilateralLayerPytorch"
    else:
      return "BilateralLayer"

  def run(self):
    output = self.op(self.image, self.guide)
    loss = output.sum()
    loss.backward()

  def reset(self):
    sz = 512
    bs = 8
    c = 16
    im = th.randn(bs, c, sz, sz)
    guide = th.rand(bs, sz, sz)
    self.image = Variable(im, requires_grad=True)
    self.guide = Variable(guide, requires_grad=True)

    if self.pytorch:
      self.op = modules.BilateralLayerTorch(c, c, 3, False)
    else:
      self.op = modules.BilateralLayer(c, c, 3, False)

    if self.cuda:
      self.image = self.image.cuda()
      self.guide = self.guide.cuda()
      self.op = self.op.cuda()


class VGG(Benchmark):
  def __init__(self, cuda=False, pytorch=False, burn_iters=5, iters=10):
    super(VGG, self).__init__(
        cuda=cuda, burn_iters=burn_iters, iters=iters)
    self.pytorch = pytorch

  def name(self):
    if self.pytorch:
      return "VGGPytorch"
    else:
      return "VGG"

  def run(self):
    output = self.op(self.image)
    loss = output.mean()
    print loss.data.cpu()[0]

  def reset(self):
    bs = 1
    self.image = Variable(th.rand(bs, 3, 224, 224), requires_grad=False)
    self.op = modules.VGG(pytorch=self.pytorch)

    if self.cuda:
      self.image = self.image.cuda()
      self.op = self.op.cuda()


class BackwardConv2d(Benchmark):
  def __init__(self, cuda=False, general_scatter=False, burn_iters=5, iters=10):
    super(BackwardConv2d, self).__init__(
        cuda=cuda, burn_iters=burn_iters, iters=iters)
    self.general_scatter = general_scatter

  def name(self):
    if self.general_scatter:
      return "BackwardConv2dGeneralScatter"
    else:
      return "BackwardConv2d"

  def run(self):
    output = self.op(self.image)

  def reset(self):
    sz = 256
    bs = 1
    c = 3
    im = th.randn(bs, c, sz, sz)
    guide = th.rand(bs, sz, sz)
    self.image = Variable(im, requires_grad=True)

    if self.general_scatter:
      self.op = modules.BackwardConv2dGeneralScatter(c, c, 3)
    else:
      self.op = modules.Conv2d(c, c, 3)

    if self.cuda:
      self.image = self.image.cuda()
      self.op = self.op.cuda()


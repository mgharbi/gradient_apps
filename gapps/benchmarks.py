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
        # print "iter {}: {:.2f}ms".format(i, runtime1)
      end = time.time()
    # print prof

    runtime = (end-start)*1000.0/self.iters

    return BenchmarkResult(self.name(), runtime, self.cuda)

  def run(self):
    """This method evaluates the critical path of the algortihm to be timed."""
    pass
    
    
class SpatialTransformer(Benchmark):
  def __init__(self, cuda=False, pytorch=False, burn_iters=5, iters=50):
    super(SpatialTransformer, self).__init__(
        cuda=cuda, burn_iters=burn_iters, iters=iters)
    self.pytorch = pytorch

  def name(self):
    if self.pytorch:
      return "SpatialTransformerPytorch"
    else:
      return "SpatialTransformer"

  def run(self):
    image = Variable(self.image, requires_grad=True)
    affine_mtx = Variable(self.affine_mtx, requires_grad=True)
    output = self.op(image, affine_mtx)
    loss = output.sum()

    loss.backward()

    # Make sure pytorch actually runs, lazy as it is
    x = image.grad.sum().cpu().data[0]
    x = affine_mtx.grad.sum().cpu().data[0]

  def reset(self):
    sz = 512
    bs = 4
    image = th.randn(bs, 16, sz, sz)

    affine_mtx = th.zeros(bs, 2, 3)
    affine_mtx[:, 0, 1] = 1.0
    affine_mtx[:, 1, 0] = 1.0

    self.image = image
    self.affine_mtx = affine_mtx

    self.op = modules.SpatialTransformer(pytorch=self.pytorch)

    if self.cuda:
      self.image = self.image.cuda()
      self.affine_mtx = self.affine_mtx.cuda()
      self.op = self.op.cuda()


class Flownet(Benchmark):
  def __init__(self, cuda=False, pytorch=False, burn_iters=5, iters=50):
    super(Flownet, self).__init__(
        cuda=cuda, burn_iters=burn_iters, iters=iters)
    self.pytorch = pytorch

  def name(self):
    if self.pytorch:
      return "FlownetPytorch"
    else:
      return "Flownet"

  def run(self):
    image = Variable(self.image, requires_grad=True)
    warp = Variable(self.warp, requires_grad=True)
    output = self.op(image, warp)
    loss = output.sum()

    loss.backward()

    # Make sure pytorch actually runs, lazy as it is
    x = image.grad.sum().cpu().data[0]
    x = warp.grad.sum().cpu().data[0]

  def reset(self):
    sz = 512
    bs = 4
    image = th.randn(bs, 16, sz, sz)
    if self.pytorch:
      warp = th.rand(bs, sz, sz, 2)
    else:
      warp = th.rand(bs, 2, sz, sz)
    self.image = image
    self.warp = warp

    self.op = modules.BilinearResampling(pytorch=self.pytorch)

    if self.cuda:
      self.image = self.image.cuda()
      self.warp = self.warp.cuda()
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
    im = Variable(self.image, requires_grad=True)
    guide = Variable(self.guide, requires_grad=True)
    output = self.op(image, guide)
    loss = output.sum()
    loss.backward()

    # Make sure pytorch actually runs, lazy as it is
    x = im.grad.sum().cpu().data[0]
    x = guide.grad.sum().cpu().data[0]

  def reset(self):
    sz = 512
    bs = 8
    c = 16
    im = th.randn(bs, c, sz, sz)
    guide = th.rand(bs, sz, sz)
    self.image = im
    self.guide = guide

    if self.pytorch:
      self.op = modules.BilateralLayerTorch(c, c, 3, False)
    else:
      self.op = modules.BilateralLayer(c, c, 3, False)

    if self.cuda:
      self.image = self.image.cuda()
      self.guide = self.guide.cuda()
      self.op = self.op.cuda()


class BilateralSliceApply(Benchmark):
  def __init__(self, cuda=False, mode="halide", burn_iters=5, iters=10):
    super(BilateralSliceApply, self).__init__(
        cuda=cuda, burn_iters=burn_iters, iters=iters)
    assert mode in ["halide", "manual", "pytorch"]
    self.mode = mode

  def name(self):
    if self.mode == "manual":
      return "BilateralSliceApplyManualCuda"
    elif self.mode == "pytorch":
      return "BilateralSliceApplyPytorch"
    else:
      return "BilateralSliceApply"

  def run(self):
    im = Variable(self.image, requires_grad=True)
    guide = Variable(self.guide, requires_grad=True)
    grid = Variable(self.grid, requires_grad=True)
    output = self.op(grid, guide, im)
    loss = output.sum()
    loss.backward()

    # Make sure pytorch actually runs, lazy as it is
    x = im.grad.sum().cpu().data[0]
    x = guide.grad.sum().cpu().data[0]

  def reset(self):
    bs = 4
    ci = 3
    co = 3
    gd = 8
    gh = 64
    gw = 64
    h = 2048
    w = 2048
    assert(w / gw == 32 and h / gh == 32)
    im = th.randn(bs, ci, h, w)
    guide = th.rand(bs, h, w)
    grid = th.rand(bs, co*(ci+1), gd, gh, gw)
    self.image = im
    self.guide = guide
    self.grid = grid

    self.op = modules.BilateralSliceApply(self.mode)

    if self.cuda:
      self.image = self.image.cuda()
      self.guide = self.guide.cuda()
      self.grid = self.grid.cuda()
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
    print(loss.data.cpu()[0])

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
    image = Variable(self.image, requires_grad=True)
    output = self.op(image)

  def reset(self):
    sz = 256
    bs = 1
    c = 3
    im = th.randn(bs, c, sz, sz)
    guide = th.rand(bs, sz, sz)
    self.image = im

    if self.general_scatter:
      self.op = modules.BackwardConv2dGeneralScatter(c, c, 3)
    else:
      self.op = modules.Conv2d(c, c, 3)

    if self.cuda:
      self.image = self.image.cuda()
      self.op = self.op.cuda()

class Demosaick(Benchmark):
  def __init__(self, cuda=False, num_filters=8, fsize=5, burn_iters=5, iters=10):
    super(Demosaick, self).__init__(
        cuda=cuda, burn_iters=burn_iters, iters=iters)
    self.num_filters = num_filters
    self.fsize = fsize

  def name(self):
    return "Demosaick {} filters,  fsize={}".format(self.num_filters, self.fsize)

  def run(self):
    mosaick = Variable(self.mosaick, requires_grad=False)
    output = self.op(mosaick)

  def reset(self):
    sz = 1024
    bs = 2
    c = 3
    mosaick = th.randn(bs, 1, sz, sz)
    self.mosaick = mosaick

    self.op = modules.LearnableDemosaick(
        num_filters=self.num_filters, fsize=self.fsize)

    if self.cuda:
      self.mosaick = self.mosaick.cuda()
      self.op = self.op.cuda()

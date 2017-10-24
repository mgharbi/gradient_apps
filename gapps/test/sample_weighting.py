import os
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler

from ..functions.sample_weighting import SampleWeighting

class SampleWeightingTest(unittest.TestCase):

  def setUp(self):
    self.debug = True

  def plot_grad_error(self, num, th):
    gerr = np.abs(num-th) / (np.abs(th)+1e-8)
    plt.subplot(1, 2, 1)
    plt.imshow(gerr)
    plt.xlabel("input")
    plt.ylabel("output")
    plt.title("relative gradient error")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.hist(np.ravel(gerr))
    plt.title("distribution of gradient errors")
    plt.colorbar()

    plt.show()

  def test_run_gpu(self):
    self._test_run(True, 1e-6)

  def test_gradients_gpu(self):
    self._test_gradients(True, 1e-6)

  # def test_gradients_cpu_float32(self):
  #   self._test_gradients(False, tf.float32, 2e-5)

  # def test_gradients_cpu_float64(self):
  #   self._test_gradients(False, tf.float64, 1e-6)
  #
  # def test_gradients_gpu_float32(self):
  #   self._test_gradients(True, tf.float32, 2e-5)

  def _test_run(self, on_gpu, tol):

    np.random.seed(0)
    bs = 2
    h = 8
    w = 8
    ci = 3
    co = 4
    spp = 4
    ksize = 3
    nsize = 5
    ncoords = 3
    ksize = 5
    nsize = 3
    nparams = 2*ncoords*co

    data_ = np.random.uniform(size=(bs, ci, h, w, spp)).astype(np.float32)
    coords_ = np.random.uniform(size=(bs, ncoords, h, w, spp)).astype(np.float32)
    params_ = np.zeros((bs, nparams, h, w)).astype(np.float32)
    params_[:, 0, ...] = 1.0
    params_[:, 2, ...] = 1.0
    params_[:, 3, ...] = 1.0
    kernels_ = np.random.uniform(size=(ksize, ksize, ci, co)).astype(np.float32)

    data = th.from_numpy(data_)
    coords = th.from_numpy(coords_)
    kernels = th.from_numpy(kernels_)
    params = th.from_numpy(params_)

    if on_gpu:
      data = data.cuda()
      coords = coords.cuda()
      params = params.cuda()
      kernels = kernels.cuda()

    data = Variable(data)
    coords = Variable(coords)
    kernels = Variable(kernels)
    params = Variable(params)

    out, out_w = SampleWeighting.apply(data, coords, params, kernels, nsize)

    eps = 1e-8
    out = out / (out_w + eps)

    # print("Testing shape match")
    assert out.shape[0] == bs
    assert out.shape[1] == co
    assert out.shape[2] == h
    assert out.shape[3] == w
    assert out_w.shape[0] == bs
    assert out_w.shape[1] == co
    assert out_w.shape[2] == h
    assert out_w.shape[3] == w

    in_ = np.mean(data_[0][0], axis=-1)
    out_ = np.squeeze(out.data.cpu().numpy()[0][0])
    out_w_ = np.squeeze(out_w.data.cpu().numpy()[0][0])

    if self.debug:
      plt.figure()
      plt.subplot(131)
      plt.title("input {}spp".format(spp))
      plt.imshow(in_)
      plt.subplot(132)
      plt.imshow(out_)
      plt.title("output")
      plt.subplot(133)
      plt.imshow(out_w_)
      plt.title("output_weights")
      plt.show()

  def _test_gradients(self, on_gpu, tol):
    np.random.seed(0)
    bs = 4
    h = 4
    w = 4
    ci = 2
    co = 3
    spp = 2
    ksize = 3
    nsize = 3
    ncoords = 3
    nparams = 2*ncoords*co

    data_ = np.random.uniform(size=(bs, ci, h, w, spp)).astype(np.float32)
    coords_ = np.random.uniform(size=(bs, ncoords, h, w, spp)).astype(np.float32)
    params_ = np.zeros((bs, nparams, h, w), dtype=np.float32)
    params_[:, 0, ...] = 1.0
    params_[:, 2, ...] = 1.0
    params_[:, 3, ...] = 1.0
    kernels_ = np.random.uniform(size=(ksize, ksize, ci, co)).astype(np.float32)

    data = th.from_numpy(data_)
    coords = th.from_numpy(coords_)
    kernels = th.from_numpy(kernels_)
    params = th.from_numpy(params_)

    if on_gpu:
      data = data.cuda()
      coords = coords.cuda()
      params = params.cuda()
      kernels = kernels.cuda()

    data = Variable(data, requires_grad=True)
    coords = Variable(coords)
    kernels = Variable(kernels, requires_grad=True)
    params = Variable(params, requires_grad=True)

    gradcheck(SampleWeighting.apply,
        (data, coords, params, kernels, nsize), eps=1e-4, atol=2e-2, rtol=2e-2,
         raise_exception=True)

  def test_performance(self):
    np.random.seed(0)
    bs = 8
    h = 64
    w = 64
    ci = 3
    co = 8
    spp = 4
    ksize = 3
    nsize = 7
    ncoords = 3
    nparams = 2*ncoords*co

    data_ = np.random.uniform(size=(bs, ci, h, w, spp)).astype(np.float32)
    coords_ = np.random.uniform(size=(bs, ncoords, h, w, spp)).astype(np.float32)
    params_ = np.zeros((bs, nparams, h, w), dtype=np.float32)
    params_[:, 0, ...] = 1.0
    params_[:, 2, ...] = 1.0
    params_[:, 3, ...] = 1.0
    kernels_ = np.random.uniform(size=(ksize, ksize, ci, co)).astype(np.float32)

    data = th.from_numpy(data_)
    coords = th.from_numpy(coords_)
    kernels = th.from_numpy(kernels_)
    params = th.from_numpy(params_)

    data = data.cuda()
    coords = coords.cuda()
    params = params.cuda()
    kernels = kernels.cuda()

    data = Variable(data, requires_grad=True)
    coords = Variable(coords)
    kernels = Variable(kernels, requires_grad=True)
    params = Variable(params, requires_grad=True)

    with profiler.profile() as prof:
      out, out_w = SampleWeighting.apply(data, coords, params, kernels, nsize)
      loss = out.sum() + out_w.sum()
      loss.backward()
    print (prof)

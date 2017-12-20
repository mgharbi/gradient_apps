# Hack to avoid launching gtk
import matplotlib 
matplotlib.use('Agg') 

import os
import time
import unittest
import skimage.io

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

# ----------- cpu/gpu test calls --------------------------------------------
def test_conv1d_cpu():
  _test_conv1d(gpu=False)

def test_conv3d_cpu():
  _test_conv3d(gpu=False)

def test_conv3d_gpu():
  _test_conv3d(gpu=True)

def test_bilateral_layer_cpu():
  _test_bilateral_layer_(gpu=False)

def test_bilateral_layer_gpu():
  _test_bilateral_layer_(gpu=True)

def test_histogram_cpu():
  _test_histogram(False)

def test_histogram_gpu():
  _test_histogram(True)

def test_soft_histogram_cpu():
  _test_soft_histogram(False)

def test_soft_histogram_gpu():
  _test_soft_histogram(True)

def test_naive_demosaick_cpu():
  _test_naive_demosaick(False)

def test_naive_demosaick_gpu():
  _test_naive_demosaick(True)

def test_learnable_demosaick_cpu():
  _test_learnable_demosaick(False)

def test_learnable_demosaick_gpu():
  _test_learnable_demosaick(True)


def test_deconv_cg_init_cpu():
  _test_deconv_cg_init(False)

def test_deconv_cg_iteration_cpu():
  _test_deconv_cg_iteration(False)

def test_profile_deconv_cg_cpu():
  _profile_deconv_cg(False)

def test_deconv_cg_cpu():
  _test_deconv_cg(False)

# ---------------------------------------------------------------------------

def _test_conv1d(gpu=False):
  bs = 16
  ci = 64
  co = 64
  kw = 5

  w = 2048

  input_grid = Variable(th.randn(bs, ci, w), requires_grad=True)
  kernels = Variable(th.randn(co, ci, kw), requires_grad=True)

  if gpu:
    input_grid = input_grid.cuda()
    kernels = kernels.cuda()

  print("profiling")
  with profiler.profile() as prof:
    for i in range(10):
      output = funcs.Conv1d.apply(
          input_grid, kernels)
      loss = output.sum()
      loss.backward()

      output2 = funcs.Conv1d.apply(
          input_grid, kernels, True)
      loss = output2.sum()
      loss.backward()

  print(prof)


def _test_conv3d(gpu=False):
  bs = 16
  ci = 8
  co = 8
  kh = 3
  kw = 3
  kd = 3

  # grid size
  d = 8
  h = 64
  w = 64

  input_grid = Variable(th.randn(bs, ci, d, h, w), requires_grad=True)
  kernels = Variable(th.randn(co, ci, kd, kh, kw), requires_grad=True)

  if gpu:
    input_grid = input_grid.cuda()
    kernels = kernels.cuda()

  print("profiling")
  with profiler.profile() as prof:
    for i in range(1):
      output = funcs.Conv3d.apply(
          input_grid, kernels)
      loss = output.sum()
      loss.backward()

  print(prof)

def _test_bilateral_layer_(gpu=False):
  bs = 1;
  ci = 3
  co = 3
  kh = 3
  kw = 3
  kd = 3

  sx, sy, sz = 4, 4, 8

  image = skimage.io.imread(os.path.join(data_dir, "rgb.png"))
  guide = skimage.io.imread(os.path.join(data_dir, "gray.png"))

  skimage.io.imsave(os.path.join(out_dir, "bilateral_layer_input.png"), image)
  sz = 256
  image = image[:sz, :sz, :]
  guide = guide[:sz, :sz]

  h, w = guide.shape
  image = np.expand_dims(image.transpose([2, 0 , 1])/255.0, 0).astype(np.float32)
  guide = np.expand_dims(guide/255.0, 0).astype(np.float32)

  image = Variable(th.from_numpy(image), requires_grad=False)
  guide = Variable(th.from_numpy(guide), requires_grad=False)
  kernels = Variable(th.randn(co, ci, kd, kh, kw), requires_grad=True)

  conv = th.nn.Conv2d(ci, co, [kh*sy, kw*sx])

  if gpu:
    image = image.cuda()
    guide = guide.cuda()
    kernels = kernels.cuda()
    conv.cuda()

  print("profiling")
  with profiler.profile() as prof:
    output = funcs.BilateralLayer.apply(
        image, guide, kernels, sx, sy, sz)
    output2 = conv(image)
    loss = output.sum()
    loss.backward()

  print(prof)

  print("testing dimensions")
  assert output.shape[0] == bs
  assert output.shape[1] == co
  assert output.shape[2] == h
  assert output.shape[3] == w

  print("testing forward")
  for i, o in enumerate([output, output2]):
    mini, maxi = o.min(), o.max()
    o -= mini
    o /= (maxi-mini)

    o = o.data[0].cpu().numpy()
    o = np.clip(np.transpose(o, [1, 2, 0]), 0, 1)
    o = np.squeeze(o)
    skimage.io.imsave(
        os.path.join(out_dir, "bilateral_layer_{}.png".format(i)), o)

  # print "testing gradient"
  # gradcheck(
  #     funcs.BilateralLayer.apply,
  #     (image, guide, kernels, sx, sy, sz),
  #     eps=1e-4, atol=5e-2, rtol=5e-4,
  #      raise_exception=True)


def _test_histogram(gpu=False):
  image = skimage.io.imread(
      os.path.join(data_dir, "gray.png")).astype(np.float32)/255.0
  image = image[:16, :16]
  nbins = 8

  image = Variable(th.from_numpy(image), requires_grad=True)
  if gpu:
    image = image.cuda()

  print("profiling")
  with profiler.profile() as prof:
    output = funcs.Histogram.apply(image, nbins)
    loss = output.sum()
    loss.backward()
  print(prof)

  print("checking gradients")
  image = image[:32, :32]
  gradcheck(funcs.Histogram.apply,
      (image, nbins), eps=1e-4, atol=5e-2, rtol=5e-4,
      raise_exception=True)


def _test_soft_histogram(gpu=False):
  image = skimage.io.imread(os.path.join(data_dir, "gray.png")).astype(np.float32)/255.0
  image = image[:64, :64]
  nbins = 8

  image = Variable(th.from_numpy(image), requires_grad=True)
  if gpu:
    image = image.cuda()

  print("profiling")
  with profiler.profile() as prof:
    output = funcs.SoftHistogram.apply(image, nbins)
    loss = output.sum()
    loss.backward()
  print(prof)


def _test_naive_demosaick(gpu=False):
  image = skimage.io.imread(
      os.path.join(data_dir, "rgb.png")).astype(np.float32)/255.0
  h, w, _ = image.shape
  mosaick = utils.make_mosaick(image)
  skimage.io.imsave(
      os.path.join(out_dir, "naive_mosaick.png"), mosaick)

  mosaick = Variable(th.from_numpy(mosaick), requires_grad=True)
  mosaick = mosaick.view(1, 1, h, w)
  if gpu:
    mosaick = mosaick.cuda()
  print("profiling")

  op = modules.NaiveDemosaick()

  with profiler.profile() as prof:
    for i in range(1):
      output = op(mosaick).view(3, h, w)
      loss = output.sum()
      loss.backward()

  # print prof

  assert output.shape[0] == 3
  assert output.shape[1] == h
  assert output.shape[2] == w

  output = output.data.cpu().numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(
      os.path.join(out_dir, "naive_demosaicked.png"), output)


def test_learnable_demosaick_gradients():
  bs = 1
  h = 64
  w = 64
  fsize = 5
  mosaick = Variable(th.randn(bs, 1, h, w))
  gfilt = Variable(th.randn(fsize), requires_grad=True)
  grad_filt = Variable(th.randn(fsize))
  print("Testing green filters grad")
  gradcheck(
      funcs.LearnableDemosaick.apply,
      (mosaick, gfilt, grad_filt),
      eps=1e-4, atol=5e-2, rtol=5e-4,
       raise_exception=True)

  print("Testing gradient filters grad")
  gfilt = Variable(th.randn(fsize))
  grad_filt = Variable(th.randn(fsize), requires_grad=True)
  gradcheck(
      funcs.LearnableDemosaick.apply,
      (mosaick, gfilt, grad_filt),
      eps=1e-4, atol=5e-2, rtol=5e-4,
       raise_exception=True)

def _test_learnable_demosaick(gpu=False):
  image = skimage.io.imread(
      os.path.join(data_dir, "rgb.png")).astype(np.float32)/255.0
  h, w, _ = image.shape
  mosaick = utils.make_mosaick(image)
  skimage.io.imsave(
      os.path.join(out_dir, "learnable_mosaick.png"), mosaick)

  mosaick = Variable(th.from_numpy(mosaick), requires_grad=True)
  mosaick = mosaick.view(1, 1, h, w)
  op = modules.LearnableDemosaick()

  if gpu:
    mosaick = mosaick.cuda()
    op.cuda()

  print("profiling")
  with profiler.profile() as prof:
    for i in range(1):
      output = op(mosaick).view(1, h, w)
      # loss = output.sum()
      # loss.backward()
  print(prof)
  print(output)

  output = output.data.cpu().numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(
      os.path.join(out_dir, "learnable_demosaicked.png"), output)

def _profile_deconv_cg(gpu=False):
  x = Variable(th.randn(1, 3, 240+11, 320+11), requires_grad=False)
  kernel = Variable(th.rand(11, 11), requires_grad=False)
  op = modules.DeconvCG()
  print("profiling")
  with profiler.profile() as prof:
    for i in range(1):
      y = op(x, kernel, 5, 10)
      loss = y.sum()
      loss.backward()
  print(prof)

def test_learnable_demosaick_cpu_gpu():
  # image = skimage.io.imread(
  #     os.path.join(data_dir, "rgb.png")).astype(np.float32)/255.0
  # h, w, _ = image.shape
  # mosaick = utils.make_mosaick(image)
  # skimage.io.imsave(
  #     os.path.join(out_dir, "learnable_mosaick.png"), mosaick)

  # mosaick = th.from_numpy(mosaick).view(1, 1, h, w)

  h = w = 128
  mosaick = th.randn(64, 1, h, w)
  mosaick = Variable(mosaick, requires_grad=True)
  op = modules.LearnableDemosaick()

  outputs = []
  grads = []
  grads2 = []
  grads3 = []
  for gpu in [False, True]:
    if gpu:
      mosaick = mosaick.cuda()
      op.cuda()

    output = op(mosaick)
    loss = output.sum()
    loss.backward()

    grads.append(op.sel_filts.grad.data.cpu().numpy().copy())
    grads2.append(op.green_filts.grad.data.cpu().numpy().copy())
    # grads3.append(mosaick.grad.data.cpu().numpy().copy())
    op.zero_grad()
    # mosaick.grad.zero_()
    outputs.append(output.data.cpu().numpy().copy())

  diff = np.abs(outputs[0]-outputs[1]).max()
  assert diff < 1e-5

  # diff = np.abs(grads3[0]-grads3[1]).max()
  # assert diff < 1e-5


  diff = np.abs(grads[0]-grads[1])
  rel_diff = diff / (np.abs(grads[0]) + 1e-5)
  rel_diff = np.abs(rel_diff).max()
  assert rel_diff < 1e-2

  diff = np.abs(grads2[0]-grads2[1])
  rel_diff = diff / (np.abs(grads2[0]) + 1e-5)
  rel_diff = np.abs(rel_diff).max()
  assert rel_diff < 1e-2

def _test_deconv_cg_init(gpu=False):
  #blurred = Variable(th.randn(1, 1, 5, 5), requires_grad=False)
  #x0 = Variable(th.randn(1, 5, 5), requires_grad=False)
  #kernel = Variable(th.rand(3, 3), requires_grad=False)
  #reg_kernel_weights = Variable(th.rand(2), requires_grad=False)
  #reg_kernels = Variable(th.randn(2, 3, 3), requires_grad=False)
  #reg_target_kernels = Variable(th.randn(2, 3, 3), requires_grad=True)
  #w_kernel = Variable(th.randn(1, 5, 5), requires_grad=False)
  #w_reg_kernels = Variable(th.randn(2, 1, 5, 5), requires_grad=False)
  blurred = Variable(th.randn(1, 1, 1, 1), requires_grad=False)
  x0 = Variable(th.randn(1, 1, 1, 1), requires_grad=True)
  kernel = Variable(th.rand(1, 1), requires_grad=False)
  reg_kernel_weights = Variable(th.rand(1), requires_grad=False)
  reg_kernels = Variable(th.randn(1, 1, 1), requires_grad=False)
  reg_target_kernels = Variable(th.randn(1, 1, 1), requires_grad=False)
  w_kernel = Variable(th.randn(1, 1, 1), requires_grad=False)
  w_reg_kernels = Variable(th.randn(1, 1, 1, 1), requires_grad=False)
  gradcheck(
      funcs.DeconvCGInit.apply,
      (blurred, x0, kernel, reg_kernel_weights, reg_kernels, reg_target_kernels, w_kernel, w_reg_kernels),
      eps=1e-6, atol=1e-5, rtol=1e-2,
       raise_exception=True)

def _test_deconv_cg_iteration(gpu=False):
  xrp = Variable(th.randn(3, 3, 16, 16), requires_grad=False)
  kernel = Variable(th.rand(7, 7), requires_grad=False)
  reg_kernel_weights = Variable(th.rand(2), requires_grad=True)
  reg_kernels = Variable(th.randn(2, 5, 5), requires_grad=True)
  gradcheck(
      funcs.DeconvCGIter.apply,
      (xrp, kernel, reg_kernel_weights, reg_kernels),
      eps=1e-4, atol=5e-3, rtol=5e-4,
       raise_exception=True)

def _test_deconv_cg(gpu=False):
  image = skimage.io.imread(
      os.path.join(data_dir, "rgb.png")).astype(np.float32)/255.0
  image = image[:244, :324, :]
  kernel = utils.sample_kernel(7)
  blurred = utils.make_blur(image, kernel)
  skimage.io.imsave(
      os.path.join(out_dir, "blurred.png"), blurred)
  blurred = blurred.transpose((2, 0, 1))

  blurred = Variable(th.from_numpy(blurred), requires_grad=False).contiguous().view(1, 3, 244, 324)
  kernel = Variable(th.from_numpy(kernel), requires_grad=False)
  op = modules.DeconvCG()

  if gpu:
    blurred = blurred.cuda()
    op.cuda()

  output = op(blurred, kernel).view(3, 244, 324)

  output = output.data.cpu().numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(
      os.path.join(out_dir, "deconv.png"), output)


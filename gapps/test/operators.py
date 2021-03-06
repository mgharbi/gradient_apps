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

def test_bilateral_grid_cpu():
  _test_bilateral_grid_(gpu=False)

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

def test_profile_deconv_cg_gpu():
  _profile_deconv_cg(True)

def test_deconv_cg_cpu():
  _test_deconv_cg(False)

def test_deconv_nonlinear_cg_cpu():
  _test_deconv_nonlinear_cg(False)

def test_deconv_nonlinear_cg_gpu():
  _test_deconv_nonlinear_cg(True)

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

  if gpu:
    image = image.cuda()
    guide = guide.cuda()
    kernels = kernels.cuda()

  print("profiling")
  with profiler.profile() as prof:
    output = funcs.BilateralLayer.apply(
        image, guide, kernels)
    loss = output.sum()
    loss.backward()

  print(prof)

  print(output.shape)

  print("testing dimensions")
  assert output.shape[0] == bs
  assert output.shape[1] == co
  assert output.shape[2] == h
  assert output.shape[3] == w

  print("testing forward")
  for i, o in enumerate([output]):
    mini, maxi = o.min(), o.max()
    o -= mini
    o /= (maxi-mini)

    o = o.data[0].cpu().numpy()
    o = np.clip(np.transpose(o, [1, 2, 0]), 0, 1)
    o = np.squeeze(o)
    skimage.io.imsave(
        os.path.join(out_dir, "bilateral_layer_{}.png".format(i)), o)

  print("testing gradient")
  gradcheck(
      funcs.BilateralLayer.apply,
      (image, guide, kernels),
      eps=1e-4, atol=5e-2, rtol=5e-4,
       raise_exception=True)

def _test_bilateral_grid_(gpu=False):
  bs = 1
  c = 3
  sigma_s, sigma_r = 4, 8

  image = skimage.io.imread(os.path.join(data_dir, "rgb.png"))
  sz = 256
  image = image[22:sz+22, 73:sz+73, :]

  h, w, _ = image.shape
  #image = np.expand_dims(image.transpose([2, 0, 1])/255.0, 0).astype(np.float32)
  image = image.transpose([2, 0, 1]).astype(np.float32)/255.0

  image = Variable(th.from_numpy(image), requires_grad=False)
  filter_s = Variable(th.zeros(5), requires_grad=True)
  filter_s.data[0] = 1.0
  filter_s.data[1] = 4.0
  filter_s.data[2] = 6.0
  filter_s.data[3] = 4.0
  filter_s.data[4] = 1.0
  filter_r = Variable(th.zeros(5), requires_grad=True)
  filter_r.data[0] = 1.0
  filter_r.data[1] = 4.0
  filter_r.data[2] = 6.0
  filter_r.data[3] = 4.0
  filter_r.data[4] = 1.0

  if gpu:
    image = image.cuda()
    filter_s = filter_s.cuda()
    filter_r = filter_r.cuda()

  output = funcs.BilateralGrid.apply(
              image, filter_s, filter_r)
  output = output.data.cpu().numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  image = image.data.cpu().numpy()
  image = np.clip(np.transpose(image, [1, 2, 0]), 0, 1)
  image = np.squeeze(image)
  skimage.io.imsave(
      os.path.join(out_dir, "bilateral_grid_input.png"), image)
  skimage.io.imsave(
      os.path.join(out_dir, "bilateral_grid_output.png"), output)
  return

  print("profiling")
  with profiler.profile() as prof:
    for i in range(20):
      output = funcs.BilateralGrid.apply(
          image, filter_s, filter_r)
      loss = output.sum()
      loss.backward()

  print(prof)

  print("testing dimensions")
  assert output.shape[0] == bs
  assert output.shape[1] == c
  assert output.shape[2] == h
  assert output.shape[3] == w

  print("testing forward")
  output = output.data[0].cpu().numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(
      os.path.join(out_dir, "bilateral_grid_output.png"), output)

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
      output = op(mosaick)
      # .view(3, h, w)
      loss = output.sum()
      loss.backward()

  print(prof)

  # assert output.shape[0] == 3
  # assert output.shape[1] == h
  # assert output.shape[2] == w
  #
  # output = output.data.cpu().numpy()
  # output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  # output = np.squeeze(output)
  # skimage.io.imsave(
  #     os.path.join(out_dir, "naive_demosaicked.png"), output)


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
      loss = output.sum()
      loss.backward()
  print(prof)
  print(output)

  output = output.data.cpu().numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(
      os.path.join(out_dir, "learnable_demosaicked.png"), output)

def _profile_deconv_cg(gpu=False):
  x = Variable(th.randn(1, 3, 256, 256), requires_grad=True)
  kernel = Variable(th.rand(11, 11), requires_grad=False)
  op = modules.DeconvCG()

  if gpu:
    x = x.cuda()
    kernel = kernel.cuda()
    op.cuda()

  print("profiling")
  it = 10
  burn_it = 5
  with profiler.profile() as prof:
    for i in range(burn_it+it):
      if i == burn_it:
        start = time.time()
      y = op(x, kernel, 1, 5)
      loss = y.sum()
      loss.backward()
    elapsed = time.time() - start
  print("elapased {:.2f} ms, {:.2f}ms/it".format(elapsed*1000, elapsed*1000.0/it))
  #print(prof)

def test_deconv_cg_match():
  x = Variable(th.randn(1, 3, 256, 256), requires_grad=True)
  kernel = Variable(th.rand(11, 11), requires_grad=False)
  op = modules.DeconvCG()

  print("CPU pass")
  y_cpu = op(x, kernel, 5, 10)

  x = x.cuda()
  kernel = kernel.cuda()
  op.cuda()

  print("GPU pass")
  y = op(x, kernel, 5, 10)

  diff = (y.cpu()-y_cpu).data.abs().max()
  print(diff)
  assert diff < 1e-2


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
  image = image[100:356, 100:356, :]
  np.random.seed(1234)
  kernel = utils.sample_psf(11)
  blurred = utils.make_blur(image, kernel)
  skimage.io.imsave(
      os.path.join(out_dir, "blurred.png"), blurred)
  blurred = blurred.transpose((2, 0, 1))

  blurred = Variable(th.from_numpy(blurred), requires_grad=False).contiguous().view(1, 3, 256, 256)
  kernel = Variable(th.from_numpy(kernel), requires_grad=False).contiguous().view(1, 11, 11)
  op = modules.DeconvCG()

  if gpu:
    blurred = blurred.cuda()
    op.cuda()

  output = op(blurred, kernel, num_irls_iter=1, num_cg_iter=20).view(3, 256, 256)
  loss = output.sum()
  loss.backward()

  output = output.data.cpu().numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(
      os.path.join(out_dir, "deconv.png"), output)

def _test_deconv_nonlinear_cg(gpu=False):
  image = skimage.io.imread(
      os.path.join(data_dir, "rgb.png")).astype(np.float32)/255.0
  w = 256
  h = 256
  image = image[100:100 + w, 100:100 + h, :]
  np.random.seed(1234)
  kernel = utils.sample_psf(11)
  blurred = utils.make_blur(image, kernel)
  skimage.io.imsave(
      os.path.join(out_dir, "blurred.png"), blurred)
  blurred = blurred.transpose((2, 0, 1))
  blurred = Variable(th.from_numpy(blurred), requires_grad=True).contiguous().view(1, 3, w, h)
  kernel = Variable(th.from_numpy(kernel), requires_grad=False).contiguous().view(1, 11, 11)
  op = modules.DeconvNonlinearCG()

  if gpu:
    blurred = blurred.cuda()
    kernel = kernel.cuda()
    op.cuda()

  output = op(blurred, kernel, num_cg_iter = 2).view(3, w, h)
  loss = output.sum()
  loss.backward()

  output = output.data.cpu().numpy()
  output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
  output = np.squeeze(output)
  skimage.io.imsave(
      os.path.join(out_dir, "deconv.png"), output)

def test_bilateral_layer_op():
  bs = 4
  h = 128
  w = 128
  ci = 3
  co = 3
  ksize = 3
  op = modules.BilateralLayerTorch(ci, co, ksize, False)
  op2 = modules.BilateralLayer(ci, co, ksize, False)
  image = Variable(th.randn(bs, ci, h, w))
  guide = Variable(th.rand(bs, h, w), requires_grad=True)

  nits_burns = 2
  nits = 5
  names = ["torch", "ours"]
  for i, o in enumerate([op, op2]):
    for it in range(nits_burns):
      output = o(image, guide)
      loss = output.sum()
      loss.backward()

    start = time.time()
    for it in range(nits):
      output = o(image, guide)
      loss = output.sum()
      loss.backward()
    end = time.time()

    print("{}: running time {}ms".format(names[i], (end-start)*1000/nits))

def test_bilateral_layer_output():
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

  op = modules.BilateralLayerTorch(3, 3, 1, False)
  op2 = modules.BilateralLayer(3, 3, 1, False)

  op.conv.weight.data.fill_(1.0)
  op2.weights.data.copy_(op.conv.weight.data)


  for i, o in enumerate([op, op2]):
    output = o(image, guide)

    mini, maxi = output.min(), output.max()
    output -= mini
    output /= (maxi-mini)

    print(mini.cpu().data[0], maxi.cpu().data[0])

    output = output.data[0].cpu().numpy()
    output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
    output = np.squeeze(output)
    skimage.io.imsave(
        os.path.join(out_dir, "bilateral_layer_{}.png".format(i)), output)

def test_stn_gpu():
  test_stn(True)

def test_stn_cpu():
  test_stn(False)

def test_stn(cuda=True):
  image = skimage.io.imread(os.path.join(data_dir, "rgb.png"))

  sz = 512
  image = image[:sz, :sz, :]

  bs = 16
  h, w, _ = image.shape
  image = np.expand_dims(
      image.transpose([2, 0 , 1])/255.0, 0).astype(np.float32)
  image = np.tile(image, [bs, 1 ,1 ,1])

  affine_mtx = th.zeros(bs, 2, 3)
  affine_mtx[:, 0, 1] = 1.0
  affine_mtx[:, 1, 0] = 1.0

  image = Variable(th.from_numpy(image), requires_grad=True)
  affine_mtx = Variable(affine_mtx, requires_grad=True)

  if cuda:
    image = image.cuda()
    affine_mtx = affine_mtx.cuda()

  nits_burns = 5
  nits = 10
  for pytorch in [False, True]:
    op = modules.SpatialTransformer(pytorch=pytorch)
    if cuda:
      op = op.cuda()

    if pytorch:
      name = "stn_torch_output.png"
    else:
      name = "stn_ours_output.png"

    for it in range(nits_burns):
      output = op(image, affine_mtx)
      loss = output.sum()
      loss.backward()

      start = time.time()
      for it in range(nits):
        output = op(image, affine_mtx)
        loss = output.sum()
        loss.backward()
      end = time.time()

    print("{}: running time {}ms".format(name, (end-start)*1000/nits))

    # output = output.data[0].cpu().numpy()
    # output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
    # output = np.squeeze(output)
    # skimage.io.imsave(
    #     os.path.join(out_dir, name), output)

def test_bilinear_resampling(cuda=True):
  image = skimage.io.imread(os.path.join(data_dir, "rgb.png"))

  sz = 256
  image = image[:sz, :sz, :]

  bs = 16
  h, w, _ = image.shape
  image = np.expand_dims(
      image.transpose([2, 0 , 1])/255.0, 0).astype(np.float32)
  image = np.tile(image, [bs, 1 ,1 ,1])

  xx, yy = np.meshgrid(np.linspace(-1, 1, sz), np.linspace(-1, 1, sz))
  xx = th.from_numpy(xx.astype(np.float32))
  yy = th.from_numpy(yy.astype(np.float32))

  dx = 0.1*th.cos(yy*2*np.pi*8.0) + yy
  dy = 0.0*th.sin(yy*2*np.pi*8.0) + xx
  dx = dx.unsqueeze(0).unsqueeze(0)
  dy = dy.unsqueeze(0).unsqueeze(0)
  warp = th.cat([dx, dy], 1)
  warp = warp.repeat(bs, 1, 1, 1)

  image = Variable(th.from_numpy(image), requires_grad=True)
  warp = Variable(warp, requires_grad=True)

  if cuda:
    image = image.cuda()
    warp = warp.cuda()

  nits_burns = 5
  nits = 20
  for mode in ["halide", "nvidia", "pytorch"]:
    if mode == "pytorch":
      w = warp.permute(0, 2, 3, 1)
    else:
      w = warp
    op = modules.BilinearResampling(mode=mode)
    if cuda:
      op = op.cuda()

    name = "resampling_"+mode+"_output.png"

    for it in range(nits_burns):
      output = op(image, w)
      loss = output.sum()
      loss.backward()

    start = time.time()
    for it in range(nits):
      output = op(image, w)
      loss = output.sum()
      loss.backward()
    end = time.time()

    print("{}: running time {}ms".format(name, (end-start)*1000/nits))

    print(output.min().data[0], output.max().data[0])

    output = output.data[0].cpu().numpy()
    output = np.clip(np.transpose(output, [1, 2, 0]), 0, 1)
    output = np.squeeze(output)
    skimage.io.imsave(
        os.path.join(out_dir, name), output)


def test_burst_demosaicking(cuda=False):
  images = []
  shift_x = [1, 0, 1, 0]
  shift_y = [1, 0, 1, 0]
  for i in range(4):
    im = skimage.io.imread(os.path.join(data_dir, "burst", "{}.png".format(i)))
    im = im[:, :, 0].astype(np.float32)/255.0  # images are grayscales
    im = im[shift_y[i]:, shift_x[i]:]
    im = np.pad(im, ((0, shift_y[i]), (0, shift_x[i])), 'constant')
    im = np.expand_dims(im, 0)
    im = th.from_numpy(im)
    images.append(im)
  images = th.cat(images, 0)
  images = Variable(images, requires_grad=True)

  n, h, w = images.shape

  demos = modules.NaiveDemosaick()


  for i in range(n):
    im = images[0].view(1, 1, h, w)
    out = demos(im)
    out = out.data[0].cpu().numpy()
    out = np.clip(np.transpose(out, [1, 2, 0]), 0, 1)
    out = np.squeeze(out)
    skimage.io.imsave(
        os.path.join(out_dir, "burst_demosaick_{}.png".format(i)), out)

  images = images.data

  # init to identity
  homographies = th.zeros(n, 8)
  homographies[:, 0] = 1.0
  homographies[:, 4] = 1.0
  homographies = homographies

  # init to first image, with naive demosaick
  init_idx = 1
  im = images[init_idx].view(1, 1, h, w)
  init = demos(im)
  recons = init.data.clone().squeeze(0)

  op = modules.BurstDemosaicking()

  gradient_weight = th.ones(1)*1e-5

  if cuda:
    op = op.cuda()
    recons = recons.cuda()
    homographies = homographies.cuda()
    images = images.cuda()
    gradient_weight = gradient_weight.cuda()

  images = Variable(images, False)
  recons = Variable(recons, True)
  gradient_weight = Variable(gradient_weight, False)
  homographies = Variable(homographies, True)

  lr = 1e1
  optimizer = th.optim.SGD(
      [{'params': [homographies], 'lr': 1e-5*lr},
        {'params': [recons], 'lr': 1e-1*lr}],
      momentum=0.9, nesterov=True)

  for step in range(1000):
    optimizer.zero_grad()
    loss, reproj = op(images, homographies, recons, gradient_weight)
    loss.backward()
    # print recons.grad.abs().max().data[0]
    # print homographies.grad.abs().max().data[0]
    optimizer.step()
    print("Step {} loss = {:.4f}".format(step, loss.data[0]))

  for i in range(n):
    print(list(homographies.cpu().data[i].numpy()))

  for i in range(n):
    im = reproj[0].view(1, 1, h, w)
    out = im.abs()*10
    out = out.data[0].cpu().numpy()
    out = np.clip(np.transpose(out, [1, 2, 0]), 0, 1)
    out = np.squeeze(out)
    skimage.io.imsave(
        os.path.join(out_dir, "burst_demosaick_reproj_error{}.png".format(i)), out)

  out = recons.data.cpu().numpy()
  out = np.clip(np.transpose(out, [1, 2, 0]), 0, 1)
  out = np.squeeze(out)
  skimage.io.imsave(
      os.path.join(out_dir, "burst_demosaicking_reconstructed.png"), out)

  diff = (init.data - recons.data.cpu()).abs().numpy()
  print("Max diff", diff.max())
  diff = np.squeeze(diff)*100
  diff = np.clip(np.transpose(diff, [1, 2, 0]), 0, 1)
  skimage.io.imsave(
      os.path.join(out_dir, "burst_demosaicking_diff_from_init.png"), diff)

def test_vgg_gpu():
  test_vgg(cuda=True)

def test_vgg_cpu():
  test_vgg(cuda=False)

def test_vgg(cuda=True):
  bs = 1
  im = th.rand(bs, 3, 224, 224)
  im = Variable(im, requires_grad=False)

  if cuda:
    im = im.cuda()

  nits_burns = 0
  nits = 1
  for pytorch in [False]:
    if pytorch:
      name = "vgg_pytorch"
      op = modules.VGG(pytorch=pytorch)
    else:
      name = "vgg_ours"
      op = modules.VGGours()
    if cuda:
      op = op.cuda()

    for it in xrange(nits_burns):
      output = op(im)

    print("running")
    start = time.time()
    # with profiler.profile() as prof:
    for it in xrange(nits):
      output = op(im)
      loss = output.mean()
      if pytorch:
        loss = output.sum()
        loss.backward()
        # loss.backward()
        # print output.cpu().data.numpy()[0, :4]
        # print "loss = {:.2f}".format(loss.data.cpu()[0])
      # th.cuda.synchronize()
    # print prof
    end = time.time()

    print("{}: running time {}ms".format(name, (end-start)*1000/nits))

def test_bilateral_slice_apply(gpu=True):
  bs = 20
  ci = 3
  co = 3
  gd = 8
  gh = 16
  gw = 16

  h = 256
  w = 256

  grid =  th.rand(bs, (ci+1)*co, gd, gh, gw)
  guide = th.rand(bs, h, w)*0.5
  input = th.rand(bs, ci, h, w)


  grid = grid.cuda()
  guide = guide.cuda()
  input = input.cuda()

  grid = Variable(grid, requires_grad=True)
  guide = Variable(guide, requires_grad=True)
  input = Variable(input, requires_grad=True)

  nbits = 5
  nits = 10

  for mode in ["pytorch", "manual", "halide"]:
    op = modules.BilateralSliceApply(mode)
    op.cuda()

    for i in range(nbits):
      out = op(grid, guide, input)
      loss = out.mean()
      loss.backward()
    start = time.time()
    for i in range(nits):
      out = op(grid, guide, input)
      loss = out.mean()
      loss.backward()
      print(loss.data.cpu()[0], input.grad.data.mean(), grid.grad.data.mean(), guide.grad.data.mean())
    # th.cuda.synchronize()
    end = time.time()
    print("{}: running time {}ms".format(mode, (end-start)*1000/nits))


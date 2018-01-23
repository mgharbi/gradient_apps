from abc import ABCMeta, abstractmethod

import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import time
import math

import gapps.functions as funcs
import gapps.resample2d_package.modules.resample2d as nvidia_resample

class NaiveDemosaick(nn.Module):
  def __init__(self):
    super(NaiveDemosaick, self).__init__()

  def forward(self, mosaick):
    output = funcs.NaiveDemosaick.apply(mosaick)
    return output

class LearnableDemosaick(nn.Module):
  def __init__(self, num_filters=8, fsize=5, sigmoid_param=1.0):
    super(LearnableDemosaick, self).__init__()

    self.num_filters = num_filters
    self.fsize = fsize

    # Register parameters that need gradients as data members
    # c, y, x order
    self.sel_filts = nn.Parameter(th.zeros(num_filters, fsize, fsize))
    self.green_filts = nn.Parameter(th.zeros(num_filters, fsize, fsize))
    self.h_chroma_filter = nn.Parameter(th.zeros(fsize, fsize))
    self.v_chroma_filter = nn.Parameter(th.zeros(fsize, fsize))
    self.q_chroma_filter = nn.Parameter(th.zeros(fsize, fsize))

    self.reset_weights()

  def reset_weights(self):
    self.sel_filts.data.normal_(0.0, 1.0)
    self.green_filts.data.normal_(1.0/(self.fsize*self.fsize), 1e-2)
    self.h_chroma_filter.data.normal_(1.0/(self.fsize*self.fsize), 1e-2)
    self.v_chroma_filter.data.normal_(1.0/(self.fsize*self.fsize), 1e-2)
    self.q_chroma_filter.data.normal_(1.0/(self.fsize*self.fsize), 1e-2)

  def forward(self, mosaick):
    output = funcs.LearnableDemosaick.apply(
        mosaick, self.sel_filts, self.green_filts,
        self.h_chroma_filter, self.v_chroma_filter, self.q_chroma_filter)
    return output

class FancyDemosaick(nn.Module):
  def __init__(self):
    super(FancyDemosaick, self).__init__()

    self.weights = [
        ("tmp", nn.Parameter(th.rand(9))),
        ]
    self.weights2d = [
        ("tmp2", nn.Parameter(th.rand(9, 8))),
        ]

    self.weights3d = [
        ("g_interp", nn.Parameter(1.0/(7*7)*th.rand(9, 7, 7))),
        ("cd_interp", nn.Parameter(0.5/(7*7)*th.rand(9, 7, 7))),
        ("cd_interp_g", nn.Parameter(0.5/(7*7)*th.rand(9, 7, 7))),
        ("fcd_interp", nn.Parameter(0.5/(7*7)*th.rand(9*2, 7, 7))),
        ("fcd_interp_g", nn.Parameter(0.5/(7*7)*th.rand(9*2, 7, 7))),
        ]

    self.weights4d = [
        ("g_weights", nn.Parameter(th.rand(9, 4, 7, 7))),
        ("cd_weights", nn.Parameter(th.rand(9, 4, 7, 7))),
        ("cdg_weights", nn.Parameter(th.rand(9, 4, 7, 7))),
        ("fcd0_weights", nn.Parameter(th.rand(9*2, 4, 7, 7))),
        ("fcd1_weights", nn.Parameter(th.rand(9*2, 4, 7, 7))),
        ("fcdg_weights", nn.Parameter(th.rand(9*2, 4, 7, 7))),
        ]

    for k, v in self.weights:
      self.register_parameter(k, v)

    for k, v in self.weights2d:
      self.register_parameter(k, v)

    for k, v in self.weights3d:
      self.register_parameter(k, v)

    for k, v in self.weights4d:
      self.register_parameter(k, v)

  def forward(self, mosaick):
    weights = [w[1] for w in self.weights]
    weights += [w[1] for w in self.weights2d]
    weights += [w[1] for w in self.weights3d]
    weights += [w[1] for w in self.weights4d]
    output = funcs.FancyDemosaick.apply(
        mosaick, *weights)
    return output

class FancyDemosaick2(nn.Module):
  def __init__(self):
    super(FancyDemosaick, self).__init__()

    self.weights = [
        ("dir_weights_x", nn.Parameter(th.rand(5))),
        ("dir_weights_y", nn.Parameter(th.rand(5))),
        ("dir_interp_g", nn.Parameter(th.rand(5))),
        ("dir_weights_n", nn.Parameter(th.rand(5))),
        ("dir_weights_p", nn.Parameter(th.rand(5))),
        ]
    self.weights2d = [
        ("neigh_weights_dx", nn.Parameter(th.rand(4, 4))),
        ("neigh_weights_dy", nn.Parameter(th.rand(4, 4))),
        ("neigh_weights_dp", nn.Parameter(th.rand(4, 4))),
        ("neigh_weights_dn", nn.Parameter(th.rand(4, 4))),
        ("diag_interp_rb", nn.Parameter(th.rand(2, 2))),
        ("cd_dir_weights_x", nn.Parameter(th.rand(2, 5))),
        ("cd_dir_weights_y", nn.Parameter(th.rand(2, 5))),
        ]

    self.weights3d = [
        ("cd_weights_dx", nn.Parameter(th.rand(2, 4, 4))),
        ("cd_weights_dy", nn.Parameter(th.rand(2, 4, 4))),
        ("dir_interp_cd", nn.Parameter(th.randn(2, 4, 5))),
        ]

    for k, v in self.weights:
      self.register_parameter(k, v)

    for k, v in self.weights2d:
      self.register_parameter(k, v)

    for k, v in self.weights3d:
      self.register_parameter(k, v)

  def forward(self, mosaick):
    weights = [w[1] for w in self.weights]
    weights += [w[1] for w in self.weights2d]
    weights += [w[1] for w in self.weights3d]
    output = funcs.FancyDemosaick.apply(
        mosaick, *weights)
    return output

class DeconvCG(nn.Module):
  def __init__(self,
               data_kernel_size=5,
               num_data_kernels=6,
               reg_kernel_size=5,
               num_reg_kernels=5,
               filter_s_size=11,
               filter_r_size=5,
               num_stages=1,
               ref=False):
    super(DeconvCG, self).__init__()
    self.num_stages = num_stages
    self.ref = ref

    # Use different kernels for different stages
    self.data_kernels = nn.Parameter(th.zeros(num_stages + 1, num_data_kernels, data_kernel_size, data_kernel_size))
    self.data_kernel_weights = nn.Parameter(th.zeros(num_stages + 1, num_data_kernels))
    self.reg_kernels = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernel_weights = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels))
    self.reg_powers = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels))
    self.filter_s = nn.Parameter(th.zeros(num_stages, filter_s_size))
    self.filter_r = nn.Parameter(th.zeros(num_stages, filter_r_size))
    self.reg_thresholds = nn.Parameter(th.zeros(num_stages, num_reg_kernels))

    assert reg_kernel_size % 2 == 1
    assert filter_s_size % 2 == 1
    assert filter_r_size % 2 == 1

    reg_kernel_center = int(reg_kernel_size / 2)
    # dx
    self.reg_kernels.data[:, 0, reg_kernel_center, reg_kernel_center] = -1.0
    self.reg_kernels.data[:, 0, reg_kernel_center + 1, reg_kernel_center] = 1.0
    # dy
    self.reg_kernels.data[:, 1, reg_kernel_center, reg_kernel_center] = -1.0
    self.reg_kernels.data[:, 1, reg_kernel_center, reg_kernel_center + 1] = 1.0
    # dxdy
    self.reg_kernels.data[:, 2, reg_kernel_center    , reg_kernel_center    ] =  1.0
    self.reg_kernels.data[:, 2, reg_kernel_center + 1, reg_kernel_center    ] = -1.0
    self.reg_kernels.data[:, 2, reg_kernel_center    , reg_kernel_center + 1] = -1.0
    self.reg_kernels.data[:, 2, reg_kernel_center + 1, reg_kernel_center + 1] =  1.0
    # d^2x
    self.reg_kernels.data[:, 3, reg_kernel_center - 1, reg_kernel_center] = 1.0
    self.reg_kernels.data[:, 3, reg_kernel_center, reg_kernel_center] = -2.0
    self.reg_kernels.data[:, 3, reg_kernel_center + 1, reg_kernel_center] = 1.0
    # d^2y
    self.reg_kernels.data[:, 4, reg_kernel_center, reg_kernel_center - 1] = 1.0
    self.reg_kernels.data[:, 4, reg_kernel_center, reg_kernel_center] = -2.0
    self.reg_kernels.data[:, 4, reg_kernel_center, reg_kernel_center + 1] = 1.0
   
    self.data_kernels.data[:, 0, reg_kernel_center, reg_kernel_center] = 1.0
    self.data_kernels.data[:, 1:, :, :] = self.reg_kernels.data[:, :, :, :]

    self.data_kernel_weights.data[:, 0] = 1.0
    self.reg_kernel_weights.data[0, :] = 0.001
    self.reg_kernel_weights.data[1:, :] = 0.05

    self.filter_s.data[:, 3] = 0.0
    self.filter_s.data[:, 4] = 1.0
    self.filter_s.data[:, 5] = 2.0
    self.filter_s.data[:, 6] = 1.0
    self.filter_s.data[:, 7] = 0.0
    self.filter_r.data[:, :] = self.filter_s.data[:, 3:8]

    self.reg_thresholds.data[:, 0] = 0.065
    self.reg_thresholds.data[:, 1] = 0.065
    self.reg_thresholds.data[:, 2] = 0.0325
    self.reg_thresholds.data[:, 3] = 0.0325
    self.reg_thresholds.data[:, 4] = 0.0325

    self.reg_powers.data[:, :] = 2.0

  def train(self, mode=True):
    super(DeconvCG, self).train(mode)
    for p in self.parameters():
      p.requires_grad = mode
    return self

  def forward(self, blurred_batch, kernel_batch, num_irls_iter, num_cg_iter, cg_tol = 1e-4):
    begin = time.time()
    num_batches = blurred_batch.shape[0]
    result = blurred_batch.new(
      blurred_batch.shape[0], blurred_batch.shape[1], blurred_batch.shape[2], blurred_batch.shape[3])
    for b in range(num_batches):
      blurred = blurred_batch[b, :, :, :]
      kernel = kernel_batch[b, :, :]
      def deconvolve(x, reg_targets, index):
        """
            Solve |\sum_{i} K * dk_i * x - dk_i * b|^2 + \sum_i |rk_i * x - reg_targets|^p_i
        """
        w_data = \
          blurred.new(self.data_kernels.shape[1], blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(1.0)
        w_reg = \
          blurred.new(self.reg_kernels.shape[1], blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(1.0)
        for irls_it in range(num_irls_iter):
          xrp = funcs.DeconvCGInit.apply(
                  blurred,
                  x,
                  kernel,
                  self.data_kernel_weights[index, :],
                  self.data_kernels[index, :, :, :],
                  self.reg_kernel_weights[index, :],
                  self.reg_kernels[index, :, :, :],
                  reg_targets,
                  w_data,
                  w_reg)
          r = xrp[1, :, :, :].norm()
          r0 = r
          for cg_it in range(num_cg_iter):
            xrp = funcs.DeconvCGIter.apply(
                    xrp,
                    kernel,
                    self.data_kernel_weights[index, :],
                    self.data_kernels[index, :, :, :],
                    self.reg_kernel_weights[index, :],
                    self.reg_kernels[index, :, :, :],
                    w_data,
                    w_reg)
            r = xrp[1, :, :, :].norm()
            if r < cg_tol * r0:
              break
    
          x = xrp[0, :, :, :]
          if (irls_it < num_irls_iter - 1):
            w_reg_kernels = funcs.DeconvCGWeight.apply(blurred, x,
               self.reg_kernels[index, :, :], reg_targets, self.reg_powers[index, :])
        return x

      # Solve the deconvolution using reg_targets == 0 with IRLS first
      reg_targets = \
        blurred.new(self.reg_kernels.shape[1], blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(0.0)
      x = blurred
      x = deconvolve(x, reg_targets, 0)
  
      for stage in range(self.num_stages):
        # Smooth out the resulting image with bilateral grid
        x = funcs.BilateralGrid.apply(x, self.filter_s[stage, :], self.filter_r[stage, :])

        # Compute the adaptive prior
        reg_targets = funcs.DeconvPrior.apply(x,
          self.reg_kernels[stage + 1, :], self.reg_thresholds[stage, :])

        x = deconvolve(x, reg_targets, stage + 1)
      result[b, :, :, :] = x
   
    assert(not np.isnan(result.data.cpu()).any())
    return result

class DeconvNonlinearCG(nn.Module):
  def __init__(self,
               data_kernel_size=5,
               num_data_kernels=6,
               reg_kernel_size=5,
               num_reg_kernels=5,
               filter_s_size=11,
               filter_r_size=5,
               num_stages=1,
               ref=False):
    super(DeconvNonlinearCG, self).__init__()
    self.num_stages = num_stages
    self.ref = ref

    # Use different kernels for first & second phases
    self.data_kernels = nn.Parameter(th.zeros(num_stages + 1, num_data_kernels, data_kernel_size, data_kernel_size))
    self.data_kernel_weights = nn.Parameter(th.zeros(num_stages + 1, num_data_kernels))
    self.reg_kernels = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernel_weights = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels))
    self.reg_powers = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels))
    self.filter_s = nn.Parameter(th.zeros(num_stages, filter_s_size))
    self.filter_r = nn.Parameter(th.zeros(num_stages, filter_r_size))
    self.reg_thresholds = nn.Parameter(th.zeros(num_stages, num_reg_kernels))

    assert reg_kernel_size % 2 == 1
    assert filter_s_size % 2 == 1
    assert filter_r_size % 2 == 1

    reg_kernel_center = int(reg_kernel_size / 2)
    # dx
    self.reg_kernels.data[:, 0, reg_kernel_center, reg_kernel_center] = -1.0
    self.reg_kernels.data[:, 0, reg_kernel_center + 1, reg_kernel_center] = 1.0
    # dy
    self.reg_kernels.data[:, 1, reg_kernel_center, reg_kernel_center] = -1.0
    self.reg_kernels.data[:, 1, reg_kernel_center, reg_kernel_center + 1] = 1.0
    # dxdy
    self.reg_kernels.data[:, 2, reg_kernel_center    , reg_kernel_center    ] =  1.0
    self.reg_kernels.data[:, 2, reg_kernel_center + 1, reg_kernel_center    ] = -1.0
    self.reg_kernels.data[:, 2, reg_kernel_center    , reg_kernel_center + 1] = -1.0
    self.reg_kernels.data[:, 2, reg_kernel_center + 1, reg_kernel_center + 1] =  1.0
    # d^2x
    self.reg_kernels.data[:, 3, reg_kernel_center - 1, reg_kernel_center] = 1.0
    self.reg_kernels.data[:, 3, reg_kernel_center, reg_kernel_center] = -2.0
    self.reg_kernels.data[:, 3, reg_kernel_center + 1, reg_kernel_center] = 1.0
    # d^2y
    self.reg_kernels.data[:, 4, reg_kernel_center, reg_kernel_center - 1] = 1.0
    self.reg_kernels.data[:, 4, reg_kernel_center, reg_kernel_center] = -2.0
    self.reg_kernels.data[:, 4, reg_kernel_center, reg_kernel_center + 1] = 1.0
   
    self.data_kernels.data[:, 0, reg_kernel_center, reg_kernel_center] = 1.0
    self.data_kernels.data[:, 1:, :, :] = self.reg_kernels.data[:, :, :, :]

    self.data_kernel_weights.data[:, 0] = 1.0
    self.reg_kernel_weights.data[0, :] = 0.001
    self.reg_kernel_weights.data[1:, :] = 0.05

    self.filter_s.data[:, 3] = 0.0
    self.filter_s.data[:, 4] = 1.0
    self.filter_s.data[:, 5] = 2.0
    self.filter_s.data[:, 6] = 1.0
    self.filter_s.data[:, 7] = 0.0
    self.filter_r.data[:, :] = self.filter_s.data[:, 3:8]

    self.reg_thresholds.data[:, 0] = 0.065
    self.reg_thresholds.data[:, 1] = 0.065
    self.reg_thresholds.data[:, 2] = 0.0325
    self.reg_thresholds.data[:, 3] = 0.0325
    self.reg_thresholds.data[:, 4] = 0.0325

    self.reg_powers.data[:, :] = 2.0

  def train(self, mode=True):
    super(DeconvNonlinearCG, self).train(mode)
    for p in self.parameters():
      p.requires_grad = mode
    return self

  def forward(self, blurred_batch, kernel_batch, num_cg_iter, cg_tol = 1e-4):
    num_batches = blurred_batch.shape[0]
    result = blurred_batch.new(
      blurred_batch.shape[0], blurred_batch.shape[1], blurred_batch.shape[2], blurred_batch.shape[3])
    for b in range(num_batches):
      blurred = blurred_batch[b, :, :, :]
      kernel = kernel_batch[b, :, :]

      # Solve the deconvolution using reg_targets == 0 with CG first
      reg_targets = \
        blurred.new(self.reg_kernels.shape[1],
                blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(0.0)
      x = blurred.clone()
      def conjugate_gradient(x, index, reg_targets):
        grad = funcs.DeconvGrad.apply(blurred, x, kernel,
          self.data_kernel_weights[index, :], self.data_kernels[index, :, :],
          self.reg_kernel_weights[index, :], self.reg_kernels[index, :, :],
          self.reg_powers[index, :], reg_targets)
        r = -grad
        r_norm = th.dot(r, r)
        r0 = r_norm
        p = r
        for cg_it in range(num_cg_iter):
          alpha = funcs.DeconvAlpha.apply(blurred, x, kernel,
            self.data_kernel_weights[index, :], self.data_kernels[index, :, :],
            self.reg_kernel_weights[index, :], self.reg_kernels[index, :, :],
            self.reg_powers[index, :], reg_targets, p)
          x = x + alpha * p
          grad = funcs.DeconvGrad.apply(blurred, x, kernel,
            self.data_kernel_weights[index, :], self.data_kernels[index, :, :],
            self.reg_kernel_weights[index, :], self.reg_kernels[index, :, :],
            self.reg_powers[index, :], reg_targets)
          r = -grad
          new_r_norm = th.dot(r, r)
          # Fletcher-Reeves update rule
          beta = new_r_norm / r_norm
          r_norm = new_r_norm
          if (r_norm < cg_tol * r0):
              break
          p = r + beta * p
        return x
      x = conjugate_gradient(x, 0, reg_targets)

      for stage in range(self.num_stages):
        # Smooth out the resulting image with bilateral grid
        x = funcs.BilateralGrid.apply(x, self.filter_s[stage, :], self.filter_r[stage, :])

        # Compute the adaptive prior
        reg_targets = funcs.DeconvPrior.apply(x,
          self.reg_kernels[stage + 1, :], self.reg_thresholds[stage, :])

        # Solve the deconvolution again using the new targets
        x = conjugate_gradient(x, stage + 1, reg_targets)

      result[b, :, :, :] = x

    assert(not np.isnan(result.data.cpu()).any())
    return result

class NonLocalMeans(nn.Module):
  def __init__(self,
               feature_filter_size=7,
               feature_channel_size=3,
               patch_filter_size=9,
               inv_sigma=1.0,
               search_radius=9):
    super(NonLocalMeans, self).__init__()

    self.feature_filter = nn.Parameter(
            th.zeros(feature_filter_size, feature_filter_size, 3, feature_channel_size))
    self.patch_filter = nn.Parameter(th.zeros(patch_filter_size, patch_filter_size))
    self.inv_sigma = nn.Parameter(th.zeros(1))
    self.search_radius = Variable(th.IntTensor([search_radius]))

    feature_filter_center = int(feature_filter_size/2)
    self.feature_filter.data[feature_filter_center, feature_filter_center, 0, 0] = 1.0
    self.feature_filter.data[feature_filter_center, feature_filter_center, 1, 1] = 1.0
    self.feature_filter.data[feature_filter_center, feature_filter_center, 2, 2] = 1.0
    self.patch_filter.data[:, :] = 1.0
    self.inv_sigma.data[0] = inv_sigma

  def train(self, mode=True):
    super(NonLocalMeans, self).train(mode)
    for p in self.parameters():
      p.requires_grad = mode
    return self

  def forward(self, input):
    output = funcs.NonLocalMeans.apply(input, self.feature_filter, self.patch_filter, self.inv_sigma, self.search_radius)
    assert(not np.isnan(output.data.cpu()).any())
    return output

class BilateralLayerBase(nn.Module):
  __metaclass__ = ABCMeta
  def __init__(self, ninputs, noutputs, kernel_size=3, use_bias=True):
    super(BilateralLayerBase, self).__init__()

    self.ninputs = ninputs
    self.noutputs = noutputs
    self.kernel_size = kernel_size
    self.use_bias = use_bias


  @abstractmethod
  def apply(self, input, guide):
    pass

  def forward(self, input, guide):
    filtered = self.apply(input, guide)
    return filtered


class BilateralLayerTorch(BilateralLayerBase):
  def __init__(self, ninputs, noutputs, kernel_size=3, use_bias=True):
    super(BilateralLayerTorch, self).__init__(
        ninputs, noutputs, kernel_size=kernel_size, use_bias=use_bias)
    self.sigma_s = 8
    self.sigma_r = 8

    self.conv = th.nn.Conv3d(
        ninputs, noutputs, self.kernel_size, bias=self.use_bias, 
        padding=self.kernel_size // 2)

    self.reset_params()

    self.is_cuda = False

  def reset_params(self):
    if self.use_bias:
      self.conv.bias.data.zero_()

  def cuda(self, device=None):
    super(BilateralLayerTorch, self).cuda(device=device)
    self.is_cuda = True
    return self

  def apply(self, input, guide):
    bs, ci, h, w = input.shape
    sigma_s = self.sigma_s
    sigma_r = self.sigma_r
    norm = 1.0/(sigma_s*sigma_s)

    guide = guide.unsqueeze(1)

    guide_pos = guide*sigma_r
    lower_bin = th.clamp(th.floor(guide_pos-0.5), min=0)
    upper_bin = th.clamp(lower_bin+1, max=sigma_r-1)
    weight = th.abs(guide_pos-0.5 - lower_bin)

    lower_bin = lower_bin.long()
    upper_bin = upper_bin.long()

    # Grid dimensions
    gw = w // sigma_s
    gh = h // sigma_s
    grid = input.new()
    grid.resize_(bs, ci, gh, gw, sigma_r)
    grid.zero_()

    # Splat
    batch_idx = th.from_numpy(np.arange(bs)).view(bs, 1, 1, 1)
    c_idx = th.from_numpy(np.arange(ci)).view(1, ci, 1, 1)
    h_idx = th.from_numpy(np.arange(h)).view(1, 1, h, 1) / sigma_s
    w_idx = th.from_numpy(np.arange(w)).view(1, 1, 1, w) / sigma_s
    if self.is_cuda:
      batch_idx = batch_idx.cuda()
      c_idx = c_idx.cuda()
      h_idx = h_idx.cuda()
      w_idx = w_idx.cuda()

    grid[batch_idx, c_idx, h_idx, w_idx, lower_bin] += (1-weight)*norm*input
    grid[batch_idx, c_idx, h_idx, w_idx, upper_bin] += weight*norm*input

    # Conv3D
    grid = self.conv(grid)

    # Slice
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    gx = th.from_numpy(((xx+0.5)/w) * gw)
    gy = th.from_numpy(((yy+0.5)/h) * gh)
    gz = guide*sigma_r

    if self.is_cuda:
      gx = gx.cuda()
      gy = gy.cuda()

    # Enclosing cell
    fx = th.floor(gx - 0.5);
    fy = th.floor(gy - 0.5);
    fz = th.clamp(th.floor(gz-0.5), min=0)

    # Trilerp weights
    wx = Variable(gx - 0.5 - fx);
    wy = Variable(gy - 0.5 - fy);
    wz = th.abs(gz-0.5 - fz)

    fx = fx.long()
    fy = fy.long()
    fz = fz.long()

    cx = th.clamp(fx+1, max=gw-1);
    cy = th.clamp(fy+1, max=gh-1);
    cz = th.clamp(fz+1, max=sigma_r-1)


    # Make indices broadcastable
    fz = fz[:, 0].view(bs, 1, h, w)
    cz = cz[:, 0].view(bs, 1, h, w)


    out = grid[batch_idx, c_idx, fy, fx, fz]*(1-wx)*(1-wy)*(1-wz) + \
          grid[batch_idx, c_idx, fy, fx, cz]*(1-wx)*(1-wy)*(  wz) + \
          grid[batch_idx, c_idx, cy, fx, fz]*(1-wx)*(  wy)*(1-wz) + \
          grid[batch_idx, c_idx, cy, fx, cz]*(1-wx)*(  wy)*(  wz) + \
          grid[batch_idx, c_idx, fy, cx, fz]*(  wx)*(1-wy)*(1-wz) + \
          grid[batch_idx, c_idx, fy, cx, cz]*(  wx)*(1-wy)*(  wz) + \
          grid[batch_idx, c_idx, cy, cx, fz]*(  wx)*(  wy)*(1-wz) + \
          grid[batch_idx, c_idx, cy, cx, cz]*(  wx)*(  wy)*(  wz)

    return out

class BilateralLayer(BilateralLayerBase):
  def __init__(self, ninputs, noutputs, kernel_size=3, use_bias=True):
    super(BilateralLayer, self).__init__(ninputs, noutputs, kernel_size=kernel_size, use_bias=use_bias)

    self.weights = nn.Parameter(th.rand(noutputs, ninputs, kernel_size, kernel_size, kernel_size))
    if self.use_bias:
      self.bias = nn.Parameter(th.zeros(noutputs))

  def apply(self, input, guide):
    filtered = funcs.BilateralLayer.apply(input, guide, self.weights)
    if self.use_bias:
      filtered = filtered + self.bias.view(1, self.noutputs, 1, 1)
    return filtered


class SpatialTransformer(nn.Module):
  def __init__(self, pytorch=False):
    super(SpatialTransformer, self).__init__()
    self.pytorch = pytorch

  def forward(self, x, affine_matrices):
    bs = x.shape[0]

    assert affine_matrices.shape[0] == bs
    assert affine_matrices.shape[1] == 2
    assert affine_matrices.shape[2] == 3

    if self.pytorch:
      flowfield = nn.functional.affine_grid(affine_matrices, x.shape)
      out = nn.functional.grid_sample(x, flowfield, 'bilinear', 'zeros')
    else:
      out = funcs.SpatialTransformer.apply(x, affine_matrices)

    return out

class BilinearResampling(nn.Module):
  def __init__(self, mode="halide"):
    super(BilinearResampling, self).__init__()
    assert mode in ["halide", "nvidia", "pytorch"]
    self.mode = mode

    if self.mode == "nvidia":
      self.op = nvidia_resample.Resample2d()

  def forward(self, x, warp):
    bs = x.shape[0]

    if self.mode == "pytorch":
      assert warp.shape[0] == bs
      assert warp.shape[1] == x.shape[2]
      assert warp.shape[2] == x.shape[3]
      assert warp.shape[3] == 2
      out = nn.functional.grid_sample(x, warp, 'bilinear', 'zeros')
    elif self.mode == "halide":
      assert warp.shape[0] == bs
      assert warp.shape[1] == 2
      assert warp.shape[2] == x.shape[2]
      assert warp.shape[3] == x.shape[3]
      out = funcs.BilinearResampling.apply(x, warp)
    else: # nvidia
      assert warp.shape[0] == bs
      assert warp.shape[1] == 2
      assert warp.shape[2] == x.shape[2]
      assert warp.shape[3] == x.shape[3]
      out = self.op(x, warp)

      print out.max()

    return out


class BurstDemosaicking(nn.Module):
  def __init__(self):
    super(BurstDemosaicking, self).__init__()

  def forward(self, inputs, homographies, reconstructed, gradient_weight):
    out, reproj_error = funcs.BurstDemosaicking.apply(
        inputs, homographies, reconstructed, gradient_weight)
    return out, reproj_error

class VGG(nn.Module):
  def __init__(self, pytorch=False):
    super(VGG, self).__init__()
    self.pytorch = pytorch

    self.wscale = 1e-3

    if self.pytorch:
      self.conv = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2, stride=2),
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2, stride=2),
          nn.Conv2d(128, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2, stride=2),
          nn.Conv2d(256, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2, stride=2),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2, stride=2),
          )
      self.fc = nn.Sequential(
          nn.Linear(512 * 7 * 7, 4096),
          nn.ReLU(inplace=True),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Linear(4096, 1000),
          nn.ReLU(inplace=True),
          )

      for n, p in self.named_parameters():
        if "weight" in n:
          p.data.fill_(self.wscale)
        elif "bias" in n:
          p.data.zero_()
    else:
      self.conv_weights = [
          # co, ci, ky, kx
          self.wscale*th.ones(64, 3, 3, 3),   # conv1_1
          self.wscale*th.ones(64, 64, 3, 3),  # conv1_2

          self.wscale*th.ones(128, 64, 3, 3),   # conv2_1
          self.wscale*th.ones(128, 128, 3, 3),  # conv2_2

          self.wscale*th.ones(256, 128, 3, 3),   # conv3_1
          self.wscale*th.ones(256, 256, 3, 3),   # conv3_2
          self.wscale*th.ones(256, 256, 3, 3),   # conv3_3

          self.wscale*th.ones(512, 256, 3, 3),   # conv4_1
          self.wscale*th.ones(512, 512, 3, 3),   # conv4_2
          self.wscale*th.ones(512, 512, 3, 3),   # conv4_3

          self.wscale*th.ones(512, 512, 3, 3),   # conv5_1
          self.wscale*th.ones(512, 512, 3, 3),   # conv5_2
          self.wscale*th.ones(512, 512, 3, 3),   # conv5_3
          ]

      self.fc_weights = [
          # co, ci
          self.wscale*th.ones(4096, 512*7*7),  # fc6
          self.wscale*th.ones(4096, 4096), # fc7
          self.wscale*th.ones(1000, 4096), # fc8
          ]

      self.biases = [
          th.zeros(64),
          th.zeros(64),

          th.zeros(128),
          th.zeros(128),

          th.zeros(256),
          th.zeros(256),
          th.zeros(256),

          th.zeros(512),
          th.zeros(512),
          th.zeros(512),

          th.zeros(512),
          th.zeros(512),
          th.zeros(512),

          th.zeros(4096),
          th.zeros(4096),
          th.zeros(1000),
          ]

  def cuda(self, device=None):
    super(VGG, self).cuda(device=device)
    if not self.pytorch:
      for i, p in enumerate(self.conv_weights):
        self.conv_weights[i] = p.cuda()

      for i, p in enumerate(self.fc_weights):
        self.fc_weights[i] = p.cuda()

      for i, p in enumerate(self.biases):
        self.biases[i] = p.cuda()

    return self

  def forward(self, input):
    if self.pytorch:
      conv = self.conv(input)
      bs, c, h, w = conv.shape
      conv = conv.view(bs, c*h*w)
      out = self.fc(conv)
    else:
      out = funcs.VGG.apply(
          input, self.conv_weights, self.fc_weights, self.biases)
    return out


class VGGours(nn.Module):
  def __init__(self, pytorch=False):
    super(VGGours, self).__init__()
    self.pytorch = pytorch

    self.wscale = 1e-3

    self.conv_weights = [
        # co, ci, ky, kx
        self.wscale*th.ones(64, 3, 3, 3),   # conv1_1
        self.wscale*th.ones(64, 64, 3, 3),  # conv1_2

        self.wscale*th.ones(128, 64, 3, 3),   # conv2_1
        self.wscale*th.ones(128, 128, 3, 3),  # conv2_2

        self.wscale*th.ones(256, 128, 3, 3),   # conv3_1
        self.wscale*th.ones(256, 256, 3, 3),   # conv3_2
        self.wscale*th.ones(256, 256, 3, 3),   # conv3_3

        self.wscale*th.ones(512, 256, 3, 3),   # conv4_1
        self.wscale*th.ones(512, 512, 3, 3),   # conv4_2
        self.wscale*th.ones(512, 512, 3, 3),   # conv4_3

        self.wscale*th.ones(512, 512, 3, 3),   # conv5_1
        self.wscale*th.ones(512, 512, 3, 3),   # conv5_2
        self.wscale*th.ones(512, 512, 3, 3),   # conv5_3
        ]

    self.fc_weights = [
        # co, ci
        self.wscale*th.ones(4096, 512*7*7),  # fc6
        self.wscale*th.ones(4096, 4096), # fc7
        self.wscale*th.ones(1000, 4096), # fc8
        ]

    self.biases = [
        th.zeros(64),
        th.zeros(64),

        th.zeros(128),
        th.zeros(128),

        th.zeros(256),
        th.zeros(256),
        th.zeros(256),

        th.zeros(512),
        th.zeros(512),
        th.zeros(512),

        th.zeros(512),
        th.zeros(512),
        th.zeros(512),

        th.zeros(4096),
        th.zeros(4096),
        th.zeros(1000),
        ]

  def cuda(self, device=None):
    super(VGGours, self).cuda(device=device)
    if not self.pytorch:
      for i, p in enumerate(self.conv_weights):
        self.conv_weights[i] = p.cuda()

      for i, p in enumerate(self.fc_weights):
        self.fc_weights[i] = p.cuda()

      for i, p in enumerate(self.biases):
        self.biases[i] = p.cuda()

    return self

  def forward(self, input):
    outs = funcs.VGGfwd_bwd.apply(
        input, self.conv_weights, self.fc_weights, self.biases)
    out = outs[0]
    grads = outs[1:]
    print("out: ", out.abs().max().data[0])
    for i, g in enumerate(grads):
      if i < len(self.conv_weights):
        name = "conv"
      elif i < len(self.conv_weights) + len(self.fc_weights) :
        name = "fc"
      else:
        name = "bias"
      print("-", name, g.abs().max().data[0])
    return out

class Conv2d(nn.Module):
  def __init__(self, n_in, n_out, ksize):
    super(Conv2d, self).__init__()
    self.weight = nn.Parameter(th.zeros(n_out, n_in, ksize, ksize))

  def forward(self, x):
    return funcs.Conv2d.apply(x, self.weight)
    

class BackwardConv2dGeneralScatter(nn.Module):
  def __init__(self, n_in, n_out, ksize):
    super(BackwardConv2dGeneralScatter, self).__init__()
    self.weight = nn.Parameter(th.zeros(n_out, n_in, ksize, ksize))

  def forward(self, x):
    return funcs.BackwardConv2dGeneralScatter.apply(x, self.weight)


class BilateralSliceApply(nn.Module):
  def __init__(self, mode="halide"):
    super(BilateralSliceApply, self).__init__()
    assert mode in ["halide", "manual", "pytorch"]
    self.mode = mode

    self.w = None
    self.h = None

    self.gw = None
    self.gh = None

  def forward(self, grid, guide, input):
    if self.mode == "manual":
      return funcs.BilateralSliceApplyManual.apply(grid, guide, input)
    elif self.mode == "halide":
      return funcs.BilateralSliceApply.apply(grid, guide, input)
    else: # pytorch
      # Get input dimensions
      bs, ci, h, w = input.shape
      _, c, gd, gh, gw = grid.shape

      # Coordinates in the fullres image
      xx = Variable(th.arange(0, w).cuda().view(1, -1).repeat(h, 1))
      yy = Variable(th.arange(0, h).cuda().view(-1, 1).repeat(1, w))

      # Spatial coordinates in the bilateral grid 
      gx = ((xx+0.5)/w) * gw
      gy = ((yy+0.5)/h) * gh
      gz = th.clamp(guide, 0.0, 1.0)*gd

      # Coordinates of the neighboring grid voxels
      fx = th.clamp(th.floor(gx - 0.5), min=0)
      fy = th.clamp(th.floor(gy - 0.5), min=0)
      fz = th.clamp(th.floor(gz-0.5), min=0)

      # Interpolation weights
      wx = gx - 0.5 - fx
      wy = gy - 0.5 - fy
      wx = wx.unsqueeze(0).unsqueeze(0)
      wy = wy.unsqueeze(0).unsqueeze(0)
      wz = th.abs(gz-0.5 - fz)
      wz = wz.unsqueeze(1)

      # Make the voxel coordinates integers to be use in slicing
      fx = fx.long().unsqueeze(0).unsqueeze(0)
      fy = fy.long().unsqueeze(0).unsqueeze(0)
      fz = fz.long()
      cx = th.clamp(fx+1, max=gw-1);
      cy = th.clamp(fy+1, max=gh-1);
      cz = th.clamp(fz+1, max=gd-1)

      # Make indices broadcastable
      fz = fz.view(bs, 1, h, w)
      cz = cz.view(bs, 1, h, w)

      # Indices to slice along the batch axis
      batch_idx = th.arange(bs).view(bs, 1, 1, 1).long().cuda()
      out = []
      # Number of output channels
      co = c // (ci+1)
      # Construct the output channels, one at a time
      for c_ in range(co):
        # Select the relevant affine coefficients in the grid
        c_idx = th.arange((ci+1)*c_, (ci+1)*(c_+1)).view(1, ci+1, 1, 1).long().cuda()
        # Slice to upsample them to full-res
        a = grid[batch_idx, c_idx, fz, fy, fx]*(1-wx)*(1-wy)*(1-wz) + \
                 grid[batch_idx, c_idx, cz, fy, fx]*(1-wx)*(1-wy)*(  wz) + \
                 grid[batch_idx, c_idx, fz, cy, fx]*(1-wx)*(  wy)*(1-wz) + \
                 grid[batch_idx, c_idx, cz, cy, fx]*(1-wx)*(  wy)*(  wz) + \
                 grid[batch_idx, c_idx, fz, fy, cx]*(  wx)*(1-wy)*(1-wz) + \
                 grid[batch_idx, c_idx, cz, fy, cx]*(  wx)*(1-wy)*(  wz) + \
                 grid[batch_idx, c_idx, fz, cy, cx]*(  wx)*(  wy)*(1-wz) + \
                 grid[batch_idx, c_idx, cz, cy, cx]*(  wx)*(  wy)*(  wz)

        # Construct the output channel as an affine combination of input channels
        o = th.sum(a[:, :-1, ...]*input, 1) + a[:, -1, ...]
        out.append(o.unsqueeze(1))
      # Assemble all the output channels in a single tensor
      out = th.cat(out, 1)
      return out

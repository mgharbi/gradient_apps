from abc import ABCMeta, abstractmethod

import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import time
import math

import gapps.functions as funcs

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

  def reset_params(self):
    if self.use_bias:
      self.conv.bias.data.zero_()

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
    grid[batch_idx, c_idx, h_idx, w_idx, lower_bin] += (1-weight)*norm*input
    grid[batch_idx, c_idx, h_idx, w_idx, upper_bin] += weight*norm*input

    # Conv3D
    grid = self.conv(grid)

    # Slice
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    gx = ((xx+0.5)/w) * gw
    gy = ((yy+0.5)/h) * gh
    gz = guide*sigma_r

    # Enclosing cell
    fx = np.floor(gx - 0.5).astype(np.int64);
    fy = np.floor(gy - 0.5).astype(np.int64);
    fz = th.clamp(th.floor(gz-0.5), min=0)
    cx = np.minimum(fx+1, gw-1);
    cy = np.minimum(fy+1, gh-1);
    cz = th.clamp(fz+1, max=sigma_r-1)

    # Trilerp weights
    wx = Variable(th.from_numpy((gx - 0.5 - fx).astype(np.float32)));
    wy = Variable(th.from_numpy((gy - 0.5 - fy).astype(np.float32)));
    wz = th.abs(gz-0.5 - fz)

    # Make indices broadcastable
    # fx = np.expand_dims(fx, 0)
    # fy = np.expand_dims(fy, 0)
    fz = fz.long()[:, 0].view(bs, 1, h, w)
    cz = cz.long()[:, 0].view(bs, 1, h, w)

    batch_idx = th.from_numpy(np.arange(bs)).view(bs, 1, 1, 1)
    c_idx = th.from_numpy(np.arange(ci)).view(1, ci, 1, 1)

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
  def __init__(self, pytorch=False):
    super(BilinearResampling, self).__init__()
    self.pytorch = pytorch

  def forward(self, x, warp):
    bs = x.shape[0]

    assert warp.shape[0] == bs
    assert warp.shape[1] == 2
    assert warp.shape[2] == x.shape[2]
    assert warp.shape[3] == x.shape[3]

    if self.pytorch:
      raise NotImplemented
    else:
      out = funcs.BilinearResampling.apply(x, warp)

    return out


class BurstDemosaicking(nn.Module):
  def __init__(self):
    super(BurstDemosaicking, self).__init__()

  def forward(self, inputs, homographies, reconstructed, gradient_weight):
    out, reproj_error = funcs.BurstDemosaicking.apply(
        inputs, homographies, reconstructed, gradient_weight)
    return out, reproj_error

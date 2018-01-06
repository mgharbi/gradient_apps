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
    # return output[:, 1:2, ...]

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
    self.sel_filts.data.normal_(-1.0, 1.0)
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

    # Use different kernels for first & second phases
    self.data_kernels = nn.Parameter(th.zeros(num_stages + 1, num_data_kernels, data_kernel_size, data_kernel_size))
    self.data_kernel_weights = nn.Parameter(th.zeros(num_stages + 1, num_data_kernels))
    self.reg_kernels = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernel_weights = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels))
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

  def train(self, mode=True):
    super(DeconvCG, self).train(mode)
    for p in self.parameters():
      p.requires_grad = mode
    return self

  def forward(self, blurred_batch, kernel_batch, num_irls_iter, num_cg_iter):
    begin = time.time()
    num_batches = blurred_batch.shape[0]
    result = blurred_batch.new(
      blurred_batch.shape[0], blurred_batch.shape[1], blurred_batch.shape[2], blurred_batch.shape[3])
    for b in range(num_batches):
      blurred = blurred_batch[b, :, :, :]
      kernel = kernel_batch[b, :, :]

      # Solve the deconvolution using reg_targets == 0 with IRLS first
      w_data = \
        blurred.new(self.data_kernels.shape[1], blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(1.0)
      w_reg = \
        blurred.new(self.reg_kernels.shape[1], blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(1.0)
      reg_targets = \
        blurred.new(self.reg_kernels.shape[1], blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(0.0)
      x = blurred
      for irls_it in range(num_irls_iter):
        xrp = funcs.DeconvCGInit.apply(
                blurred,
                x,
                kernel,
                self.data_kernel_weights[0, :],
                self.data_kernels[0, :, :, :],
                self.reg_kernel_weights[0, :],
                self.reg_kernels[0, :, :, :],
                reg_targets,
                w_data,
                w_reg)
        r = xrp[1, :, :, :].norm()
        if r.data.cpu() < 1e-5:
          break
        for cg_it in range(num_cg_iter):
          xrp = funcs.DeconvCGIter.apply(
                  xrp,
                  kernel,
                  self.data_kernel_weights[0, :],
                  self.data_kernels[0, :, :, :],
                  self.reg_kernel_weights[0, :],
                  self.reg_kernels[0, :, :, :],
                  w_data,
                  w_reg)
          r = xrp[1, :, :, :].norm()
          if r.data.cpu() < 1e-5:
            break
  
        x = xrp[0, :, :, :]
        #if (irls_it < num_irls_iter - 1):
        #  w_reg_kernels = funcs.DeconvCGWeight.apply(blurred, x,
        #    self.reg_kernels0, reg_targets, self.reg_powers0)
  
      for stage in range(self.num_stages):
        # Smooth out the resulting image with bilateral grid
        x = funcs.BilateralGrid.apply(x, self.filter_s[stage, :], self.filter_r[stage, :])

        # Compute the adaptive prior
        reg_targets = funcs.DeconvPrior.apply(x,
          self.reg_kernels[stage + 1, :], self.reg_thresholds[stage, :])

        # Solve the deconvolution again using the obtained reg_targets
        for irls_it in range(num_irls_iter):
          xrp = funcs.DeconvCGInit.apply(
                  blurred,
                  x,
                  kernel,
                  self.data_kernel_weights[stage + 1, :],
                  self.data_kernels[stage + 1, :, :, :],
                  self.reg_kernel_weights[stage + 1, :],
                  self.reg_kernels[stage + 1, :, :],
                  reg_targets,
                  w_data,
                  w_reg)
          r = xrp[1, :, :, :].norm()
          if r.data.cpu() < 1e-6:
            break
 
          for cg_it in range(num_cg_iter):
            xrp = funcs.DeconvCGIter.apply(
                    xrp,
                    kernel,
                    self.data_kernel_weights[stage + 1, :],
                    self.data_kernels[stage + 1, :, :, :],
                    self.reg_kernel_weights[stage + 1, :],
                    self.reg_kernels[stage + 1, :],
                    w_data,
                    w_reg)
            r = xrp[1, :, :, :].norm()
            if r.data.cpu() < 1e-5:
              break
  
          x = xrp[0, :, :, :]
          #if (irls_it < num_irls_iter):
          #  w_reg_kernels = funcs.DeconvCGWeight.apply(blurred, x,
          #    self.reg_kernels1, reg_targets, self.reg_powers1)

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

    self.feature_filter = nn.Parameter(th.zeros(feature_filter_size, feature_filter_size, 3, feature_channel_size))
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
    return output


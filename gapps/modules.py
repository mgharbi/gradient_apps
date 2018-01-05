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
    self.sel_filts.data.uniform_(-1.0, 1.0)
    self.green_filts.data.normal_(0.0, 1.0/(self.fsize*self.fsize))
    self.h_chroma_filter.data.normal_(0.0, 1.0/(self.fsize*self.fsize))
    self.v_chroma_filter.data.normal_(0.0, 1.0/(self.fsize*self.fsize))
    self.q_chroma_filter.data.normal_(0.0, 1.0/(self.fsize*self.fsize))
    self.h_chroma_filter.data.normal_(0.0, 1.0/(self.fsize*self.fsize))
    # self.green_filts.data /= self.fsize

  # def cuda(self, device=None):
  #   # self.mask = self.mask.cuda()
  #   return super(LearnableDemosaick, self).cuda(device)

  def forward(self, mosaick):
    output = funcs.LearnableDemosaick.apply(
        mosaick, self.sel_filts, self.green_filts,
        self.h_chroma_filter, self.v_chroma_filter, self.q_chroma_filter)
    # return output[:, 1:2, ...]
    return output


class DeconvCG(nn.Module):
  def __init__(self,
               precond_kernel_size=11,
               data_kernel_size=5,
               num_data_kernels=8,
               reg_kernel_size=5,
               num_reg_kernels=16,
               filter_s_size=11,
               filter_r_size=5,
               num_stages=1,
               num_gmm=3,
               ref=False):
    super(DeconvCG, self).__init__()
    self.num_stages = num_stages
    self.ref = ref

    # Use different kernels for first & second phases
    self.data_kernels = nn.Parameter(th.zeros(num_stages + 1, num_data_kernels, data_kernel_size, data_kernel_size))
    self.data_kernel_weights = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels))
    self.reg_kernels = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernel_weights = nn.Parameter(th.zeros(num_stages + 1, num_reg_kernels))
    self.precond_kernel = nn.Parameter(th.zeros(num_stages + 1, precond_kernel_size, precond_kernel_size))
    self.filter_s = nn.Parameter(th.zeros(num_stages, filter_s_size))
    self.filter_r = nn.Parameter(th.zeros(num_stages, filter_r_size))
    self.reg_thresholds = nn.Parameter(th.zeros(num_stages, num_reg_kernels))
    self.gmm_weights = nn.Parameter(th.zeros(num_stages, num_gmm, num_reg_kernels))
    self.gmm_invvars = nn.Parameter(th.zeros(num_stages, num_gmm, num_reg_kernels))

    assert precond_kernel_size % 2 == 1
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
    
    self.data_kernels.data[:, :, :, :] = 0.0
    self.data_kernels.data[:, 0, reg_kernel_center, reg_kernel_center] = 1.0

    if not ref:
      self.reg_kernels.data[:, 5:, :, :].normal_(0, 0.1)
      self.data_kernels.data[:, 1:, :, :].normal_(0, 0.1)

    self.data_kernel_weights.data[:, :] = 1.0
    self.reg_kernel_weights.data[:, :] = 0.0001
    if ref:
      # Don't use the extra kernels for reference
      self.data_kernel_weights.data[:, 1:] = 0.0
      self.reg_kernel_weights.data[:, 5:] = 0.0

    # Initialize the preconditioning kernel to a Dirac
    precond_kernel_center = int(precond_kernel_size / 2)
    self.precond_kernel.data[:, precond_kernel_center, precond_kernel_center] = 1.0

    self.filter_s.data[:, 3] = 1.0
    self.filter_s.data[:, 4] = 4.0
    self.filter_s.data[:, 5] = 6.0
    self.filter_s.data[:, 6] = 4.0
    self.filter_s.data[:, 7] = 1.0
    self.filter_r.data[:, :] = self.filter_s.data[:, 3:8]

    self.reg_thresholds.data[:, 0] = 0.065
    self.reg_thresholds.data[:, 1] = 0.065
    self.reg_thresholds.data[:, 2] = 0.0325
    self.reg_thresholds.data[:, 3] = 0.0325
    self.reg_thresholds.data[:, 4] = 0.0325
    self.reg_thresholds.data[:, 5:] = 0.01
    if not ref:
      self.reg_thresholds.data.uniform_(0, 0.03)
      self.reg_thresholds.data += 0.02

    self.gmm_weights.data[:, 0, :] = 0.30471011
    self.gmm_weights.data[:, 1, :] = 0.43436355
    self.gmm_weights.data[:, 2, :] = 0.26092634

    self.gmm_invvars.data[:, 0, :] = 7021.6804986
    self.gmm_invvars.data[:, 1, :] = 471.84142102
    self.gmm_invvars.data[:, 2, :] = 41.84820868

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
        if r.data.cpu() < 1e-6:
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
          if r.data.cpu() < 1e-6:
            break
        x = xrp[0, :, :, :]
        if (irls_it < num_irls_iter - 1):
          w_reg = funcs.DeconvCGWeight.apply(blurred, x,
            self.reg_kernels[0, :, :, :], reg_targets, self.gmm_weights[0, :, :], self.gmm_invvars[0, :, :])
 
      result[b, :, :, :] = x
  
    assert(not np.isnan(result.data.cpu()).any())
    return result


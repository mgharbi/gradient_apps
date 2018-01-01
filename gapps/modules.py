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
    return output[:, 1:2, ...]

class LearnableDemosaick(nn.Module):
  def __init__(self, num_filters=8, fsize=5, sigmoid_param=1.0):
    super(LearnableDemosaick, self).__init__()

    self.num_filters = num_filters
    self.fsize = fsize

    # Register parameters that need gradients as data members
    # c, y, x order
    self.sel_filts = nn.Parameter(th.zeros(num_filters, fsize, fsize))
    self.green_filts = nn.Parameter(th.zeros(num_filters, fsize, fsize))
    self.softmax_scale = nn.Parameter(1.0*th.ones(1, 1, 1))

    self.reset_weights()

  def reset_weights(self):
    self.sel_filts.data.uniform_(-1.0, 1.0)
    self.green_filts.data.uniform_(0.0, 1.0)
    # self.green_filts.data[:, ::2, ::2] = 0
    # self.green_filts.data[:, 1::2, 1::2] = 0
    # self.sel_filts.data[:, ::2, ::2] = 0
    # self.sel_filts.data[:, 1::2, 1::2] = 0

    # self.sel_filts.data[0, self.fsize//2, self.fsize//2-1] = -1
    # self.sel_filts.data[0, self.fsize//2, self.fsize//2+1] = 1
    # self.sel_filts.data[1, self.fsize//2-1, self.fsize//2] = -1
    # self.sel_filts.data[1, self.fsize//2+1, self.fsize//2] = 1

    # self.green_filts.data[0, self.fsize//2, self.fsize//2-1] = 0.5
    # self.green_filts.data[0, self.fsize//2, self.fsize//2+1] = 0.5
    # self.green_filts.data[1, self.fsize//2-1, self.fsize//2] = 0.5
    # self.green_filts.data[1, self.fsize//2+1, self.fsize//2] = 0.5
    # self.green_filts.data[...] = 1.0
    
    # only weigh green values
    # self.sel_filts.data[:, ::2, ::2] = 0
    # self.sel_filts.data[:, 1::2, 1::2] = 0

    # mask = th.ones_like(self.green_filts.data[0:1, ...])
    # mask[:, ::2, ::2] = 0
    # mask[:, 1::2, 1::2] = 0
    # self.register_buffer("mask", mask)

  # def cuda(self, device=None):
  #   # self.mask = self.mask.cuda()
  #   return super(LearnableDemosaick, self).cuda(device)

  def forward(self, mosaick):
    # Normalize green average
    # gfilts = []
    # sfilts = []
    # gg = self.green_filts
    # ss = self.sel_filts
    # for k in range(self.num_filters):
    #   m = Variable(self.mask)
    #   g = gg[k:k+1, ...]*m
    #   s = ss[k:k+1, ...]*m
    #   # g = g / g.sum()
    #   gfilts.append(g)
    #   sfilts.append(s)
    # gfilts = th.cat(gfilts, 0)
    # sfilts = th.cat(sfilts, 0)
    #
    # Zero sum for the selectors
    # for k in range(self.num_filters):
    #   sfilts.append(self.sel_filts[k:k+1, ...] - self.sel_filts[k:k+1, ...].sum())
    # sfilts = th.cat(sfilts, 0)*self.softmax_scale
    # sfilts = self.sel_filts
    # sfilts = self.sel_filts*self.softmax_scale

    # sel_filts = self.sel_filts*Variable(self.softmax_scale, requires_grad=False)
    output = funcs.LearnableDemosaick.apply(mosaick, self.sel_filts, self.green_filts)
    return output[:, 1:2, ...]


class DeconvCG(nn.Module):
  def __init__(self,
               precond_kernel_size=11,
               reg_kernel_size=5,
               num_reg_kernels=5,
               filter_s_size=5,
               filter_r_size=5,
               ref=False):
    super(DeconvCG, self).__init__()

    # Use different kernels for first & second phases
    self.reg_kernels0 = nn.Parameter(th.zeros(num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernels1 = nn.Parameter(th.zeros(num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernel_weights0 = nn.Parameter(th.zeros(num_reg_kernels))
    self.reg_kernel_weights1 = nn.Parameter(th.zeros(num_reg_kernels))
    self.reg_powers0 = nn.Parameter(th.zeros(num_reg_kernels))
    self.reg_powers1 = nn.Parameter(th.zeros(num_reg_kernels))
    self.precond_kernel0 = nn.Parameter(th.zeros(precond_kernel_size, precond_kernel_size))
    self.precond_kernel1 = nn.Parameter(th.zeros(precond_kernel_size, precond_kernel_size))
    self.filter_s = nn.Parameter(th.zeros(filter_s_size))
    self.filter_r = nn.Parameter(th.zeros(filter_r_size))
    self.reg_thresholds = nn.Parameter(th.zeros(num_reg_kernels))

    assert reg_kernel_size % 2 == 1

    # if not ref:
    #  self.reg_kernels.data.normal_(0, 0.1)
    #  self.reg_kernel_weights.data.normal_(0, 0.1)
    #  self.reg_powers.data.normal_(1.0, 0.02)
    #  self.precond_kernel.data.normal_(0, 0.1)

    # Initialize to L2 norm
    self.reg_powers0.data[:] = 2.0
    self.reg_powers1.data[:] = 2.0

    reg_kernel_center = int(reg_kernel_size / 2)
    # dx
    self.reg_kernels0.data[0, reg_kernel_center, reg_kernel_center] = -1.0
    self.reg_kernels0.data[0, reg_kernel_center + 1, reg_kernel_center] = 1.0
    # dy
    self.reg_kernels0.data[1, reg_kernel_center, reg_kernel_center] = -1.0
    self.reg_kernels0.data[1, reg_kernel_center, reg_kernel_center + 1] = 1.0
    # dxdy
    self.reg_kernels0.data[2, reg_kernel_center    , reg_kernel_center    ] =  1.0
    self.reg_kernels0.data[2, reg_kernel_center + 1, reg_kernel_center    ] = -1.0
    self.reg_kernels0.data[2, reg_kernel_center    , reg_kernel_center + 1] = -1.0
    self.reg_kernels0.data[2, reg_kernel_center + 1, reg_kernel_center + 1] =  1.0
    # d^2x
    self.reg_kernels0.data[3, reg_kernel_center - 1, reg_kernel_center] = 1.0
    self.reg_kernels0.data[3, reg_kernel_center, reg_kernel_center] = -2.0
    self.reg_kernels0.data[3, reg_kernel_center + 1, reg_kernel_center] = 1.0
    # d^2y
    self.reg_kernels0.data[4, reg_kernel_center, reg_kernel_center - 1] = 1.0
    self.reg_kernels0.data[4, reg_kernel_center, reg_kernel_center] = -2.0
    self.reg_kernels0.data[4, reg_kernel_center, reg_kernel_center + 1] = 1.0
    self.reg_kernels1.data = self.reg_kernels0.data.clone()

    # Smaller lambda for first phase since the noise is going to be smoothed out anyway
    self.reg_kernel_weights0.data[:] = math.sqrt(0.001)
    self.reg_kernel_weights1.data[:] = math.sqrt(0.05)

    # Initialize the preconditioning kernel to a Dirac
    precond_kernel_center = int(precond_kernel_size / 2)
    self.precond_kernel0.data[precond_kernel_center, precond_kernel_center] = 1.0
    self.precond_kernel1.data[precond_kernel_center, precond_kernel_center] = 1.0

    self.filter_s.data[0] = 0.0
    self.filter_s.data[1] = 0.0
    self.filter_s.data[2] = 1.0
    self.filter_s.data[3] = 0.0
    self.filter_s.data[4] = 0.0
    self.filter_r.data = self.filter_s.data.clone()

    self.reg_thresholds.data[0] = 0.065
    self.reg_thresholds.data[1] = 0.065
    self.reg_thresholds.data[2] = 0.0325
    self.reg_thresholds.data[3] = 0.0325
    self.reg_thresholds.data[4] = 0.0325

  def forward(self, blurred_batch, kernel_batch, num_irls_iter, num_cg_iter):
    num_batches = blurred_batch.shape[0]
    result = blurred_batch.new(
      blurred_batch.shape[0], blurred_batch.shape[1], blurred_batch.shape[2], blurred_batch.shape[3])
    for b in range(num_batches):
      blurred = blurred_batch[b, :, :, :]
      kernel = kernel_batch[b, :, :]
      # Solve the deconvolution using reg_targets == 0 with IRLS first
      w_kernel = blurred.new(blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(1.0)
      w_reg_kernels = \
        blurred.new(self.reg_kernels0.shape[0], blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(1.0)
      reg_targets = \
        blurred.new(self.reg_kernels0.shape[0], blurred.shape[0], blurred.shape[1], blurred.shape[2]).fill_(0.0)
      x = blurred.clone()
#      for irls_it in range(num_irls_iter):
#        xrp = funcs.DeconvCGInit.apply(blurred, x, kernel,
#                self.reg_kernel_weights0, self.reg_kernels0, reg_targets,
#                self.precond_kernel0, w_kernel, w_reg_kernels)
#        r = xrp[1, :, :, :].norm()
#        if r.data.cpu() < 1e-10:
#          break
#  
#        for cg_it in range(num_cg_iter):
#          xrp = funcs.DeconvCGIter.apply(xrp, kernel,
#                  self.reg_kernel_weights0, self.reg_kernels0,
#                  self.precond_kernel0, w_kernel, w_reg_kernels)
#          r = xrp[1, :, :, :].norm()
#          if r.data.cpu() < 1e-10:
#            break
#  
#        x = xrp[0, :, :, :]
#        if (irls_it < num_irls_iter - 1):
#          w_reg_kernels = funcs.DeconvCGWeight.apply(blurred, x,
#            self.reg_kernels0, reg_targets, self.reg_powers0)
  
      # Smooth out the resulting image with bilateral grid
      x = funcs.BilateralGrid.apply(x, self.filter_s, self.filter_r)
#  
#      # Compute the adaptive prior
#      reg_targets = funcs.DeconvPrior.apply(x, self.reg_kernels1, self.reg_thresholds)
#
#      # Solve the deconvolution again using the obtained reg_targets
#      for irls_it in range(num_irls_iter):
#        xrp = funcs.DeconvCGInit.apply(blurred, x, kernel,
#                self.reg_kernel_weights1, self.reg_kernels1, reg_targets,
#                self.precond_kernel0, w_kernel, w_reg_kernels)
#        r = xrp[1, :, :, :].norm()
#        if r.data.cpu() < 1e-10:
#          break
#  
#        for cg_it in range(num_cg_iter):
#          xrp = funcs.DeconvCGIter.apply(xrp, kernel,
#                  self.reg_kernel_weights1, self.reg_kernels1,
#                  self.precond_kernel1, w_kernel, w_reg_kernels)
#          r = xrp[1, :, :, :].norm()
#          if r.data.cpu() < 1e-10:
#            break
#  
#        x = xrp[0, :, :, :].clone()
#        if (irls_it < num_irls_iter):
#          w_reg_kernels = funcs.DeconvCGWeight.apply(blurred, x,
#            self.reg_kernels1, reg_targets, self.reg_powers1)
      result[b, :, :, :] = x
  
    assert(not np.isnan(result.data.cpu()).any())
    return result


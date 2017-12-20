import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

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
    # self.register_buffer("softmax_scale", th.ones(1, 1, 1))
    self.softmax_scale = nn.Parameter(1.0*th.ones(1, 1, 1))

    self.reset_weights()

  def reset_weights(self):
    self.sel_filts.data.uniform_(-1.0, 1.0)
    self.green_filts.data.uniform_(0.0, 1.0)
    self.green_filts.data[:, ::2, ::2] = 0
    self.green_filts.data[:, 1::2, 1::2] = 0
    self.sel_filts.data[:, ::2, ::2] = 0
    self.sel_filts.data[:, 1::2, 1::2] = 0

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

    mask = th.ones_like(self.green_filts.data[0:1, ...])
    mask[:, ::2, ::2] = 0
    mask[:, 1::2, 1::2] = 0
    self.register_buffer("mask", mask)

  def cuda(self, device=None):
    self.mask = self.mask.cuda()
    return super(LearnableDemosaick, self).cuda(device)

  def forward(self, mosaick):
    # Normalize green average
    gfilts = []
    sfilts = []
    gg = self.green_filts
    ss = self.sel_filts
    for k in range(self.num_filters):
      m = Variable(self.mask)
      g = gg[k:k+1, ...]*m
      s = ss[k:k+1, ...]*m
      # g = g / g.sum()
      gfilts.append(g)
      sfilts.append(s)
    gfilts = th.cat(gfilts, 0)
    sfilts = th.cat(sfilts, 0)

    # Zero sum for the selectors
    # for k in range(self.num_filters):
    #   sfilts.append(self.sel_filts[k:k+1, ...] - self.sel_filts[k:k+1, ...].sum())
    # sfilts = th.cat(sfilts, 0)*self.softmax_scale
    # sfilts = self.sel_filts
    # sfilts = self.sel_filts*self.softmax_scale

    # sel_filts = self.sel_filts*Variable(self.softmax_scale, requires_grad=False)
    output = funcs.LearnableDemosaick.apply(mosaick, sfilts, gfilts)
    return output[:, 1:2, ...]


class DeconvCG(nn.Module):
  def __init__(self, reg_kernel_size=5, num_reg_kernels=5, ref=False):
    super(DeconvCG, self).__init__()

    self.reg_kernels = nn.Parameter(th.zeros(num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernel_weights = nn.Parameter(th.zeros(num_reg_kernels))
    self.reg_target_kernels = nn.Parameter(th.zeros(num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_powers = nn.Parameter(th.zeros(num_reg_kernels))

    assert reg_kernel_size % 2 == 1

    self.reg_powers.data[:] = 1.0
    if not ref:
      self.reg_kernels.data.normal_(0, 0.1)
      self.reg_kernel_weights.data.normal_(0, 0.1)
      #self.reg_target_kernels.data.normal_(0, 0.1)
      self.reg_powers.data.normal_(1.0, 0.02)

    self.reg_kernels.data[0, 2, 2] += -2.0
    self.reg_kernels.data[0, 3, 2] += 1.0
    self.reg_kernels.data[0, 2, 3] += 1.0

    self.reg_kernel_weights.data[0] += 1.0

  def forward(self, blurred, kernel, num_irls_iter, num_cg_iter):
    w_kernel = Variable(th.ones(blurred.shape[1], blurred.shape[2], blurred.shape[3]))
    w_reg_kernels = Variable(
      th.ones(self.reg_kernels.shape[0], blurred.shape[1], blurred.shape[2], blurred.shape[3]))
    x0 = blurred.clone()
    for irls_it in range(num_irls_iter):
      xrp = funcs.DeconvCGInit.apply(blurred, x0, kernel,
              self.reg_kernel_weights, self.reg_kernels, self.reg_target_kernels, w_kernel, w_reg_kernels).clone()
      assert(not np.isnan(xrp.data).any())
      r = np.linalg.norm(xrp.data.numpy()[1, :, :, :])
      if r < 1e-10:
        break

      for cg_it in range(num_cg_iter):
        xrp = funcs.DeconvCGIter.apply(xrp, kernel,
                self.reg_kernel_weights, self.reg_kernels, w_kernel, w_reg_kernels).clone()
        assert(not np.isnan(xrp.data).any())
        r = np.linalg.norm(xrp.data.numpy()[1, :, :, :])
        if r < 1e-10:
            break
      x0 = xrp[0, :, :, :].clone()
      w_reg_kernels = funcs.DeconvCGWeight.apply(blurred, x0,
        self.reg_kernels, self.reg_target_kernels, self.reg_powers)
      assert(not np.isnan(w_reg_kernels.data).any())
    return x0

# class CG(nn.Module):
#   def forward(self, A, b):
#     r = 0
#     x = 0
#     for nit:
#       r, x, p = funcs.cg_it(r, x, p)
#     return x

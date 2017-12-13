import torch as th
import torch.nn as nn

import gapps.functions as funcs


class NaiveDemosaick(nn.Module):
  def __init__(self):
    super(NaiveDemosaick, self).__init__()

  def forward(self, mosaick):
    output = funcs.NaiveDemosaick.apply(mosaick)
    return output

class LearnableDemosaick(nn.Module):
  def __init__(self, gfilt_size=9, grad_filt_size=9):
    super(LearnableDemosaick, self).__init__()

    # Register parameters that need gradients as data members
    self.gfilt = nn.Parameter(th.zeros(gfilt_size))
    self.grad_filt = nn.Parameter(th.zeros(grad_filt_size))

    assert grad_filt_size % 2 == 1
    assert gfilt_size % 2 == 1
    assert grad_filt_size >= 3
    assert gfilt_size >= 3

    # Initialize to reference method
    self.gfilt.data.normal_(0, 1e-3)
    self.grad_filt.data.normal_(0, 1e-3)
    self.gfilt.data[gfilt_size//2 - 1] += 0.5
    self.gfilt.data[gfilt_size//2 + 1] += 0.5
    self.grad_filt.data[grad_filt_size//2 - 1] += -1
    self.grad_filt.data[grad_filt_size//2 + 1] += 1

  def forward(self, mosaick):
    output = funcs.LearnableDemosaick.apply(mosaick, self.gfilt, self.grad_filt)
    return output

class DeconvCG(nn.Module):
  def __init__(self, reg_kernel_size=3, num_reg_kernels=2):
    super(DeconvCG, self).__init__()

    self.reg_kernels = nn.Parameter(th.zeros(num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernel_weights = nn.Parameter(th.zeros(num_reg_kernels))

    assert reg_kernel_size % 2 == 1

    self.reg_kernels.data.normal_(0, 1.0)
    self.reg_kernel_weights.data.uniform_(1e-6, 1.0)

  def forward(self, image, kernel):
    xrp = funcs.DeconvCGInit.apply(image, image, kernel, self.reg_kernels, self.reg_kernel_weights)
    for iter in range(100):
      xrp = funcs.DeconvCGIter.apply(xrp, kernel, self.reg_kernels, self.reg_kernel_weights)

# class CG(nn.Module):
#   def forward(self, A, b):
#     r = 0
#     x = 0
#     for nit:
#       r, x, p = funcs.cg_it(r, x, p)
#     return x

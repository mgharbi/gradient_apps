import torch as th
import torch.nn as nn
import numpy as np

import gapps.functions as funcs

class NaiveDemosaick(nn.Module):
  def __init__(self):
    super(NaiveDemosaick, self).__init__()

  def forward(self, mosaick):
    output = funcs.NaiveDemosaick.apply(mosaick)
    return output[:, 1:2, ...]

class LearnableDemosaick(nn.Module):
  def __init__(self, num_filters=8, fsize=5):
    super(LearnableDemosaick, self).__init__()

    self.num_filters = num_filters
    self.fsize = fsize

    # Register parameters that need gradients as data members
    # c, y, x order
    self.sel_filts = nn.Parameter(th.zeros(num_filters, fsize, fsize))
    self.green_filts = nn.Parameter(th.zeros(num_filters, fsize, fsize))

    self.sel_filts.data.normal_(0, 1.0/(fsize*fsize))
    self.green_filts.data.normal_(0, 1.0/(fsize*fsize))

  def forward(self, mosaick):
    output = funcs.LearnableDemosaick.apply(mosaick, self.sel_filts, self.green_filts)
    return output[:, 1:2, ...]


class DeconvCG(nn.Module):
  def __init__(self, reg_kernel_size=3, num_reg_kernels=2):
    super(DeconvCG, self).__init__()

    self.reg_kernels = nn.Parameter(th.zeros(num_reg_kernels, reg_kernel_size, reg_kernel_size))
    self.reg_kernel_weights = nn.Parameter(th.zeros(num_reg_kernels))

    assert reg_kernel_size % 2 == 1

    self.reg_kernels.data.normal_(0, 0.01)
    self.reg_kernel_weights.data.normal_(0, 0.01)
    self.reg_kernel_weights.data += 1.0

  def forward(self, image, kernel):
    xrp = funcs.DeconvCGInit.apply(image, image, kernel, self.reg_kernel_weights, self.reg_kernels)
    #print(np.linalg.norm(xrp.data.numpy()[1, :, :, :]))
    for it in range(100):
      xrp = funcs.DeconvCGIter.apply(xrp, kernel, self.reg_kernel_weights, self.reg_kernels)
      #print(np.linalg.norm(xrp.data.numpy()[1, :, :, :]))
    return xrp[0, :, :, :]

# class CG(nn.Module):
#   def forward(self, A, b):
#     r = 0
#     x = 0
#     for nit:
#       r, x, p = funcs.cg_it(r, x, p)
#     return x

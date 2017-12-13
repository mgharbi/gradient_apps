import torch as th
import torch.nn as nn

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
    self.sel_filts = nn.Parameter(th.zeros(fsize, fsize, num_filters))
    self.green_filts = nn.Parameter(th.zeros(fsize, fsize, num_filters))

    self.sel_filts.data.normal_(0, 1.0/(fsize*fsize))
    self.green_filts.data.normal_(0, 1.0/(fsize*fsize))

  def forward(self, mosaick):
    output = funcs.LearnableDemosaick.apply(mosaick, self.sel_filts, self.green_filts)
    return output[:, 1:2, ...]


# class CG(nn.Module):
#   def forward(self, A, b):
#     r = 0
#     x = 0
#     for nit:
#       r, x, p = funcs.cg_it(r, x, p)
#     return x

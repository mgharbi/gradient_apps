import torch as th
import torch.nn as nn
import numpy as np

class GreenLoss(th.nn.Module):
  def __init__(self):
    super(GreenLoss, self).__init__()

  def forward(self, src, tgt):
    diff = src - tgt
    mse = th.mean(th.pow(diff[:, 1, ...], 2))
    return mse

class CroppedMSELoss(th.nn.Module):
  def __init__(self, crop=5):
    super(CroppedMSELoss, self).__init__()
    self.crop = crop
    self.mse = th.nn.MSELoss()

  def forward(self, src, tgt):
    crop = self.crop
    if crop > 0:
      src = src[..., crop:-crop, crop:-crop]
      tgt = tgt[..., crop:-crop, crop:-crop]

    return self.mse(src, tgt)

class PSNR(th.nn.Module):
  def __init__(self, crop=5):
    super(PSNR, self).__init__()
    self.crop = crop

  def forward(self, src, tgt):
    mse = th.mean(th.mean(th.mean(th.pow(src-tgt, 2), -1), -1), -1)
    psnr = -10*th.log(mse)/np.log(10.0)
    psnr = psnr.mean()
    return psnr

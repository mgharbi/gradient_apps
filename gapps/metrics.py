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


class CroppedL1Loss(th.nn.Module):
  def __init__(self, crop=5):
    super(CroppedL1Loss, self).__init__()
    self.crop = crop
    self.l1 = th.nn.L1Loss()

  def forward(self, src, tgt):
    crop = self.crop
    if crop > 0:
      src = src[..., crop:-crop, crop:-crop]
      tgt = tgt[..., crop:-crop, crop:-crop]

    return self.l1(src, tgt)


class CroppedGradientLoss(th.nn.Module):
  def __init__(self, crop=5):
    super(CroppedGradientLoss, self).__init__()
    self.crop = crop
    self.l1 = th.nn.L1Loss()

  def forward(self, src, tgt):
    crop = self.crop
    if crop > 0:
      src = src[..., crop:-crop, crop:-crop]
      tgt = tgt[..., crop:-crop, crop:-crop]

    dx_src = src[:, :, 1:, ...] - src[:, :, :-1, ...]
    dy_src = src[:, 1:, ...] - src[:, :-1, ...]

    dx_tgt = tgt[:, :, 1:, ...] - tgt[:, :, :-1, ...]
    dy_tgt = tgt[:, 1:, ...] - tgt[:, :-1, ...]

    dx_diff = self.l1(dx_src, dx_tgt)
    dy_diff = self.l1(dy_src, dy_tgt)
    return dx_diff + dy_diff


class PSNR(th.nn.Module):
  def __init__(self, crop=5):
    super(PSNR, self).__init__()
    self.crop = crop

  def forward(self, src, tgt):
    mse = th.mean(th.mean(th.mean(th.pow(src-tgt, 2), -1), -1), -1)
    psnr = -10*th.log(mse+1e-12)/np.log(10.0)
    psnr = psnr.mean()
    return psnr

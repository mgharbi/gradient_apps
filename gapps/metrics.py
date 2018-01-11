import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from math import exp

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


def gaussian(window_size, sigma):
  gauss = th.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
  return gauss/gauss.sum()


def create_window(window_size, sigma, channel):
  _1D_window = gaussian(window_size, sigma).unsqueeze(1)
  _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
  window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
  return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
  mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
  mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

  mu1_sq = mu1.pow(2)
  mu2_sq = mu2.pow(2)
  mu1_mu2 = mu1*mu2

  sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
  sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
  sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

  C1 = 0.01**2
  C2 = 0.03**2

  ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

  if size_average:
    return ssim_map.mean()
  else:
    return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
  def __init__(self, window_size = 11, sigma=1.5, size_average = True):
    super(SSIM, self).__init__()
    self.window_size = window_size
    self.size_average = size_average
    self.channel = 1
    self.sigma = sigma
    self.window = create_window(window_size, self.sigma, self.channel)

  def forward(self, img1, img2):
    (_, channel, _, _) = img1.size()

    if channel == self.channel and self.window.data.type() == img1.data.type():
      window = self.window
    else:
      window = create_window(self.window_size, self.sigma, channel)
      
      if img1.is_cuda:
          window = window.cuda(img1.get_device())
      window = window.type_as(img1)
      
      self.window = window
      self.channel = channel

    return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class MSSSIM(nn.Module):
  def __init__(self, window_size=11, sigmas=[0.5, 1, 2, 4, 8], size_average = True):
    super(MSSSIM, self).__init__()
    self.SSIMs = [SSIM(window_size, s, size_average=size_average) for s in sigmas]

  def forward(self, img1, img2):
    loss = 1
    for s in self.SSIMs:
      loss *= s(img1, img2)
    return loss


def ssim(img1, img2, window_size = 11, sigma=1.5, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, sigma, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def accuracy(batch_data, pred):
  (imgs, segs, infos) = batch_data
  _, preds = th.max(pred.data.cpu(), dim=1)
  valid = (segs >= 0)
  acc = 1.0 * th.sum(valid * (preds == segs)) / (th.sum(valid) + 1e-10)
  return acc, th.sum(valid)

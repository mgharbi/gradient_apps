import torch as th
import torch.nn as nn
import torch.nn.functional as F

import rendernett.modules.preprocessors as pre
import rendernett.modules.operators as ops


class LinearChain(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=32, depth=3):
    super(LinearChain, self).__init__()
    layers = [
        nn.Conv2d(ninputs, width, ksize),
        nn.ReLU(inplace=True),
        ]
    for d in range(depth-1):
      layers.append(nn.Conv2d(width, width, ksize))
      layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(width, noutputs, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


class SkipAutoencoderDownsample(nn.Module):
  def __init__(self, ninputs, ksize, width, batchnorm=True):
    super(SkipAutoencoderDownsample, self).__init__()
    if batchnorm:
      self.layer = nn.Sequential(
          nn.Conv2d(ninputs, width, ksize, stride=2, padding=ksize//2, bias=False),
          nn.BatchNorm2d(width),
          nn.ReLU(inplace=True))
    else:
      self.layer = nn.Sequential(
          nn.Conv2d(ninputs, width, ksize, stride=2, padding=ksize//2),
          nn.ReLU(inplace=True))

  def forward(self, x):
    return self.layer(x)


class SkipAutoencoderUpsample(nn.Module):
  def __init__(self, ninputs, ksize, width, batchnorm=True):
    super(SkipAutoencoderUpsample, self).__init__()
    if batchnorm:
      self.layer = nn.Sequential(
          nn.Conv2d(ninputs, width, ksize, padding=ksize//2, bias=False),
          nn.BatchNorm2d(width),
          nn.ReLU(inplace=True),
          nn.UpsamplingBilinear2d(scale_factor=2))
    else:
      self.layer = nn.Sequential(
          nn.Conv2d(ninputs, width, ksize, padding=ksize//2),
          nn.ReLU(inplace=True),
          nn.UpsamplingBilinear2d(scale_factor=2))

  def forward(self, x, prev):
    c = th.cat((self.layer(x), prev), 1)
    return c


class SkipAutoencoder(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=32, depth=3, max_width=512, 
               batchnorm=True):
    super(SkipAutoencoder, self).__init__()
    ds_layers = []
    widths = []

    w = width
    widths.append(w)
    ds_layers.append(SkipAutoencoderDownsample(ninputs, ksize, w, batchnorm=False))
    for d in range(depth-1):
      prev_w = w
      w = min(max_width, w*2)
      widths.append(w)
      ds_layers.append(SkipAutoencoderDownsample(prev_w, ksize, w, batchnorm=batchnorm))

    us_layers = []
    for d in range(depth-1, 0, -1):
      w = widths[d]
      if d < depth-1:
        w *= 2
      w2 = widths[d-1]
      us_layers.append(SkipAutoencoderUpsample(w, ksize, w2, batchnorm=batchnorm))

    self.ds_layers = nn.ModuleList(ds_layers)
    self.us_layers = nn.ModuleList(us_layers)
    self.prediction = nn.Conv2d(2*widths[0], noutputs, 1)

  def forward(self, x):
    data = []
    for l in self.ds_layers:
      x = l(x)
      data.append(x)

    x = data.pop()
    for l in self.us_layers:
      prev = data.pop()
      x = l(x, prev)

    x = self.prediction(x)
    return x

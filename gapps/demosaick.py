import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import torchlib.viz as viz
import torchlib.utils as utils

import gapps.datasets as datasets
import gapps.modules as models
import gapps.metrics as metrics


def get_pca_filters(dataset, tsize=3):
  assert tsize % 2 == 1
  mean = np.zeros((tsize*tsize,))
  covar = np.zeros((tsize*tsize, tsize*tsize))
  tot_count = 0
  for item, d in enumerate(dataset):
    m = np.squeeze(d[0])
    h, w = m.shape
    irange = range(tsize//2+1, h-tsize//2, tsize)
    jrange = range(tsize//2+1, w-tsize//2, tsize)

    data = np.zeros((len(irange)*len(jrange), tsize*tsize))
    count = 0
    for i in irange:
      for j in jrange:
        block = m[i-tsize//2:i+tsize//2+1, j-tsize//2:j+tsize//2+1]
        data[count, :] = np.ravel(block)
        count += 1
    tot_count += count
    mean += data.sum(0)
    covar += data.T.dot(data)
    if item == 20:
      break
  mean /= tot_count
  covar /= tot_count
  covar -= mean.T.dot(mean)
  covar *= tot_count*1.0/(tot_count-1)

  vals, vects = np.linalg.eig(covar)

  vects = vects.T
  vects = vects.reshape(tsize*tsize, tsize, tsize).astype(np.float32)
  return vects


class DemosaickCallback(object):
  def __init__(self, model, reference, num_batches, val_loader, env=None):
    self.model = model
    self.reference = reference
    self.num_batches = num_batches
    self.val_loader = val_loader

    self.viz = viz.BatchVisualizer("demosaick", env=env)
    self.sel_kernels = viz.BatchVisualizer("selection_kernels", env=env)
    self.green_kernels = viz.BatchVisualizer("green_kernels", env=env)

    self.green_scalar = viz.ScalarVisualizer(
        "green_distribution",
        opts={"legend": ["min", "-std", "mean", "+std", "max"]},
        env=env)

    self.sel_scalar = viz.ScalarVisualizer(
        "select_distribution",
        opts={"legend": ["min", "-std", "mean", "+std", "max"]},
        env=env)

    self.h_scalar = viz.ScalarVisualizer(
        "chroma_h",
        opts={"legend": ["min", "-std", "mean", "+std", "max"]},
        env=env)

    self.loss_viz = viz.ScalarVisualizer(
        "loss", opts={"legend": ["train", "val"]}, env=env)
    self.psnr_viz = viz.ScalarVisualizer(
        "psnr", opts={"legend": ["train", "val"]}, env=env)
    self.ssim_viz = viz.ScalarVisualizer(
        "1-ssim", opts={"legend": ["train", "val"]}, env=env)
    self.l1_viz = viz.ScalarVisualizer(
        "l1", opts={"legend": ["train", "val"]}, env=env)

    self.current_epoch = 0

  def _get_im_batch(self):
    for b in self.val_loader:
      batchv = Variable(b[0])
      if next(self.model.parameters()).is_cuda:
        batchv = batchv.cuda()
      out = self.model(batchv)
      out = out.data.cpu().numpy()
      ref = self.reference(batchv.cpu())
      ref = ref.data.cpu().numpy()
      # ref = np.zeros_like(out)

      inp = b[0].cpu().numpy()
      gt = b[1].cpu().numpy()
      diff = np.abs(gt-out)*4
      inp = np.tile(inp, [1, gt.shape[1], 1, 1])
      batchviz = np.concatenate([inp, gt, out, ref, diff], axis=0)
      batchviz = np.clip(batchviz, 0, 1)
      return batchviz

  def on_epoch_begin(self, epoch):
    self.current_epoch = epoch

  def on_epoch_end(self, epoch, logs):
    if "loss" in logs.keys():
      self.loss_viz.update(epoch+1, logs['loss'], name="val")
    if "psnr" in logs.keys():
      self.psnr_viz.update(epoch+1, logs['psnr'], name="val")
    if "ssim" in logs.keys():
      self.ssim_viz.update(epoch+1, logs['psnr'], name="val")
    if "l1" in logs.keys():
      self.l1_viz.update(epoch+1, logs['l1'], name="val")

    self.viz.update(self._get_im_batch(), per_row=self.val_loader.batch_size,
                    caption="input | gt | ours | ref | diff (x4)")

    k = self.model.sel_filts.data.clone().cpu().view(
        self.model.num_filters, 1, self.model.fsize, self.model.fsize)
    maxi = np.abs(k).max()
    k /= maxi
    k += 1
    k /= 2
    self.sel_kernels.update(k, caption="{:.2f}".format(maxi))

    k = self.model.green_filts.data.clone().cpu().view(
        self.model.num_filters, 1, self.model.fsize, self.model.fsize)
    mini, maxi = k.min(), k.max()
    k -= mini
    k /= maxi-mini
    self.green_kernels.update(k, caption="{:.2f} {:2f}".format(mini, maxi))


  def on_batch_end(self, batch, logs):
    frac = self.current_epoch + batch*1.0/self.num_batches
    if "loss" in logs.keys():
      self.loss_viz.update(frac, logs['loss'], name="train")
    if "psnr" in logs.keys():
      self.psnr_viz.update(frac, logs['psnr'], name="train")
    if "ssim" in logs.keys():
      self.ssim_viz.update(frac, logs['ssim'], name="train")
    if "l1" in logs.keys():
      self.l1_viz.update(frac, logs['l1'], name="train")

    # k = self.model.green_filts.data.clone().cpu()
    # mini, maxi = k.min(), k.max()
    # mean, std = k.mean(), k.std()
    # self.green_scalar.update(frac, mini, name="min")
    # self.green_scalar.update(frac, mean-std, name="-std")
    # self.green_scalar.update(frac, mean, name="mean")
    # self.green_scalar.update(frac, mean+std, name="+std")
    # self.green_scalar.update(frac, maxi, name="max")
    #
    # k = self.model.sel_filts.data.clone().cpu()
    # mini, maxi = k.min(), k.max()
    # mean, std = k.mean(), k.std()
    # self.sel_scalar.update(frac, mini, name="min")
    # self.sel_scalar.update(frac, mean-std, name="-std")
    # self.sel_scalar.update(frac, mean, name="mean")
    # self.sel_scalar.update(frac, mean+std, name="+std")
    # self.sel_scalar.update(frac, maxi, name="max")
    #
    # k = self.model.h_chroma_filter.data.clone().cpu()
    # mini, maxi = k.min(), k.max()
    # mean, std = k.mean(), k.std()
    # self.h_scalar.update(frac, mini, name="min")
    # self.h_scalar.update(frac, mean-std, name="-std")
    # self.h_scalar.update(frac, mean, name="mean")
    # self.h_scalar.update(frac, mean+std, name="+std")
    # self.h_scalar.update(frac, maxi, name="max")

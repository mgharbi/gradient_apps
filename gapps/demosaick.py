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

class DemosaickCallback(object):
  def __init__(self, model, reference, num_batches, val_loader, env=None):
    self.model = model
    self.reference = reference
    self.num_batches = num_batches
    self.val_loader = val_loader

    self.viz = viz.BatchVisualizer("demosaick", env=env)

    self.loss_viz = viz.ScalarVisualizer(
        "loss", opts={"legend": ["train", "val"]}, env=env)
    self.psnr_viz = viz.ScalarVisualizer(
        "psnr", opts={"legend": ["train", "train_g", "val"]}, env=env)
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

    self.show_val_batch()

  def show_val_batch(self):
    self.viz.update(self._get_im_batch(), per_row=self.val_loader.batch_size,
                    caption="input | gt | ours | ref | diff (x4)")


  def on_batch_end(self, batch, logs):
    frac = self.current_epoch + batch*1.0/self.num_batches
    if "loss" in logs.keys():
      self.loss_viz.update(frac, logs['loss'], name="train")
    if "psnr" in logs.keys():
      self.psnr_viz.update(frac, logs['psnr'], name="train")
    if "psnr_g" in logs.keys():
      self.psnr_viz.update(frac, logs['psnr_g'], name="train_g")
    if "ssim" in logs.keys():
      self.ssim_viz.update(frac, logs['ssim'], name="train")
    if "l1" in logs.keys():
      self.l1_viz.update(frac, logs['l1'], name="train")

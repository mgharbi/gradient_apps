import argparse
import logging
import os
import setproctitle
import time

import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchsample.modules as ts
import torchsample.callbacks as callbacks
import torchsample.metrics as metrics
from torchvision import transforms

import torchlib.viz as viz

import gapps.datasets as datasets
import gapps.modules as models

log = logging.getLogger("gapps")

# class PSNR(metrics.Metric):
#   def __init__(self):
#     self.sum = 0.
#     self.total_count = 0.
#
#     def reset(self):
#       self.sum = 0.
#       self.total_count = 0.
#
#     def __call__(self, y_pred, y_true=None):
#       mse = th.square(y_pred-y_true)
#       self.sum += mse
#       self.total_count +=


class DemosaickCallback(callbacks.Callback):

  def __init__(self, loader, env=None):
    super(DemosaickCallback, self).__init__()
    self.loader = loader
    self.viz = viz.BatchVisualizer("demosaick", env=env)
    self.gkernel = viz.ScalarVisualizer("green_kernel", env=env)
    self.grad_kernel = viz.ScalarVisualizer("grad_kernel", env=env)

    self.current_epoch = 0

  def on_train_begin(self, logs):
    self.train_logs = logs

  def on_epoch_begin(self, epoch, logs=None):
    self.current_epoch = epoch

  def on_batch_end(self, batch, logs=None):
    mdl = self.trainer.model
    frac = self.current_epoch + batch*1.0/self.train_logs['num_batches']
    for n, p in self.trainer.model.named_parameters():
      if n == "gfilt":
        self.gkernel.update(frac, list(p.data.cpu().numpy()))
      elif n == "grad_filt":
        self.grad_kernel.update(frac, list(p.data.cpu().numpy()))

    for b in self.loader:
      batchv = Variable(b[0])
      out = mdl(batchv)
      out = out.data.cpu().numpy()
      inp = b[0].cpu().numpy()
      gt = b[1].cpu().numpy()
      batchviz = np.concatenate([np.tile(inp, [1, 3, 1, 1]), gt, out], axis=0)
      batchviz = np.clip(batchviz, 0, 1)
      self.viz.update(batchviz, per_row=gt.shape[0], caption="input | gt | ours")
      break


  def on_epoch_end(self, epoch, logs=None):
    pass


class GreenLoss(th.nn.Module):
  def __init__(self):
    super(GreenLoss, self).__init__()

  def forward(self, src, tgt):
    diff = src - tgt
    mse = th.mean(th.pow(diff[:, 1, ...], 2))
    return mse


def main(args):
  model = models.LearnableDemosaick()

  dset = datasets.DemosaickingDataset(args.dataset, transform=datasets.ToTensor())
  val_dset = datasets.DemosaickingDataset(args.val_dataset, transform=datasets.ToTensor())
  print "Training on {} with {} images".format(args.dataset, len(dset))
  print "Validating on {} with {} images".format(args.val_dataset, len(dset))

  loader = DataLoader(dset, batch_size=args.batch_size, num_workers=4)
  val_loader = DataLoader(val_dset, batch_size=args.batch_size)

  if not os.path.exists(args.output):
    os.makedirs(args.output)
  cbks = [
      callbacks.ModelCheckpoint(args.output),
      callbacks.ExperimentLogger(args.output),
      callbacks.CSVLogger(os.path.join(args.output, "current.csv")),
      DemosaickCallback(val_loader)
      ]

  trainer = ts.ModuleTrainer(model)
  trainer.compile(
      # loss=GreenLoss(), 
      loss=th.nn.MSELoss(), 
      # optimizer=th.optim.LBFGS(model.parameters(), lr=1),
      optimizer=th.optim.Adam(model.parameters(), lr=args.lr),
      callbacks=cbks
      )

  trainer.fit_loader(loader, val_loader=val_loader, num_epoch=10, cuda_device=-1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="data/demosaick/val/filelist.txt")
  parser.add_argument("--val_dataset", default="data/demosaick/val/filelist.txt")
  parser.add_argument("--output", default="output/demosaick")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--batch_size", type=int, default=16)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps')

  main(args)

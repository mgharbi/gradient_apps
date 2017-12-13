import argparse
import logging
import os
import setproctitle
import time
import gc

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


log = logging.getLogger("gapps_demosaicking")


class DemosaickCallback(object):
  def __init__(self, model, reference, num_batches, val_loader, env=None):
    self.model = model
    self.reference = reference
    self.num_batches = num_batches
    self.val_loader = val_loader

    self.viz = viz.BatchVisualizer("demosaick", env=env)
    self.gkernel = viz.ScalarVisualizer("green_kernel", env=env)
    self.grad_kernel = viz.ScalarVisualizer("grad_kernel", env=env)

    self.loss_viz = viz.ScalarVisualizer("loss", env=env)
    self.psnr_viz = viz.ScalarVisualizer("psnr", env=env)
    self.val_loss_viz = viz.ScalarVisualizer("val_loss", env=env)
    self.val_psnr_viz = viz.ScalarVisualizer("val_psnr", env=env)

    self.current_epoch = 0

  def _get_im_batch(self):
    for b in self.val_loader:
      batchv = Variable(b[0])
      out = self.model(batchv)
      out = out.data.cpu().numpy()
      ref = self.reference(batchv)
      ref = ref.data.cpu().numpy()

      inp = b[0].cpu().numpy()
      gt = b[1].cpu().numpy()
      diff = np.abs(gt-out)*4
      batchviz = np.concatenate([np.tile(inp, [1, 3, 1, 1]), gt, out, ref, diff], axis=0)
      batchviz = np.clip(batchviz, 0, 1)
      return batchviz

  def on_epoch_begin(self, epoch):
    self.current_epoch = epoch

  def on_epoch_end(self, epoch, logs):
    if "loss" in logs.keys():
      self.val_loss_viz.update(epoch, logs['loss'])
    if "psnr" in logs.keys():
      self.val_psnr_viz.update(epoch, logs['psnr'])

    self.viz.update(self._get_im_batch(), per_row=self.val_loader.batch_size,
                    caption="input | gt | ours | ref | diff (x4)")

    for n, p in self.model.named_parameters():
      if n == "gfilt":
        self.gkernel.update(epoch, list(p.data.cpu().numpy()))
      elif n == "grad_filt":
        self.grad_kernel.update(epoch, list(p.data.cpu().numpy()))

  def on_batch_end(self, batch, logs):
    frac = self.current_epoch + batch*1.0/self.num_batches
    if "loss" in logs.keys():
      self.loss_viz.update(frac, logs['loss'])
    if "psnr" in logs.keys():
      self.psnr_viz.update(frac, logs['psnr'])


def main(args):
  model = models.LearnableDemosaick()
  reference_model = models.NaiveDemosaick()

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  dset = datasets.DemosaickingDataset(
      args.dataset, transform=datasets.ToTensor())
  val_dset = datasets.DemosaickingDataset(
      args.val_dataset, transform=datasets.ToTensor())
  log.info("Training on {} with {} images".format(
    args.dataset, len(dset)))
  log.info("Validating on {} with {} images".format(
    args.val_dataset, len(val_dset)))

  loader = DataLoader(dset, batch_size=args.batch_size, num_workers=4, shuffle=True)
  val_loader = DataLoader(val_dset, batch_size=args.batch_size)

  optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
  loss_fn = metrics.CroppedMSELoss(crop=5)
  psnr_fn = metrics.PSNR(crop=5)

  checkpointer = utils.Checkpointer(args.output, model, optimizer, verbose=False)
  callback = DemosaickCallback(model, reference_model, len(loader), val_loader, env="gapps_demosaick")

  smooth_loss = 0
  smooth_psnr = 0
  ema = 0.99
  for epoch in range(args.num_epochs):
    # Training
    model.train(True)
    with tqdm(total=len(loader), unit=' batches') as pbar:
      pbar.set_description("Epoch {}/{}".format(epoch+1, args.num_epochs))
      callback.on_epoch_begin(epoch)
      for batch_id, batch in enumerate(loader):
        mosaick, reference = batch
        mosaick = Variable(mosaick, requires_grad=False)
        reference = Variable(reference, requires_grad=False)

        output = model(mosaick)

        optimizer.zero_grad()
        loss = loss_fn(output, reference)
        loss.backward()
        optimizer.step()

        psnr = psnr_fn(output, reference)

        smooth_loss = ema*smooth_loss + (1-ema)*loss.data[0]
        smooth_psnr = ema*smooth_psnr + (1-ema)*psnr.data[0]

        logs = {"loss": smooth_loss, "psnr": smooth_psnr}
        pbar.set_postfix(logs)
        pbar.update(1)
        callback.on_batch_end(batch_id, logs)
    model.train(False)

    # Validation
    with tqdm(total=len(val_loader), unit=' batches') as pbar:
      pbar.set_description("Epoch {}/{} (val)".format(epoch+1, args.num_epochs))
      total_loss = 0
      total_psnr = 0
      n_seen = 0
      for batch_id, batch in enumerate(val_loader):
        mosaick, reference = batch
        mosaick = Variable(mosaick, requires_grad=False)
        reference = Variable(reference, requires_grad=False)

        output = model(mosaick)
        loss = loss_fn(output, reference)
        psnr = psnr_fn(output, reference)

        total_loss += loss.data[0]*args.batch_size
        total_psnr += psnr.data[0]*args.batch_size
        n_seen += args.batch_size
        pbar.update(1)

      total_loss /= n_seen
      total_psnr /= n_seen
      logs = {"loss": total_loss, "psnr": total_psnr}
      pbar.set_postfix(logs)
      callback.on_epoch_end(epoch, logs)

    # save
    checkpointer.on_epoch_end(epoch)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="data/demosaick/val/filelist.txt")
  parser.add_argument("--val_dataset", default="data/demosaick/val/filelist.txt")
  parser.add_argument("--output", default="output/demosaick")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--num_epochs", type=int, default=5)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_demosaicking')

  main(args)

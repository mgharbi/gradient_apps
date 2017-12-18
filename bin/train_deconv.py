# Hack to avoid launching gtk
import matplotlib 
matplotlib.use('Agg') 

import argparse
import logging
import os
import setproctitle
import time
import gc

import numpy as np
import scipy
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

log = logging.getLogger("gapps_deconvolution")

viz_port = 8888

class DeconvCallback(object):
  def __init__(self, model, num_batches, val_loader, env=None):
    self.model = model
    self.val_loader = val_loader
    self.num_batches = num_batches

    self.viz = viz.BatchVisualizer("deconv", port=viz_port, env=env)
    self.psf_viz = viz.BatchVisualizer("psf", port=viz_port, env=env)

    self.loss_viz = viz.ScalarVisualizer("loss", port=viz_port, env=env)
    self.psnr_viz = viz.ScalarVisualizer("psnr", port=viz_port, env=env)
    self.val_loss_viz = viz.ScalarVisualizer("val_loss", port=viz_port, env=env)
    self.val_psnr_viz = viz.ScalarVisualizer("val_psnr", port=viz_port, env=env)

    self.current_epoch = 0

  def _get_im_batch(self):
    for b in self.val_loader:
      batchv = Variable(b[0])
      psf = Variable(b[2])
      out = self.model(batchv, psf)
      out = out.data.cpu().numpy()

      inp = b[0].cpu().numpy()
      gt = b[1].cpu().numpy()
      diff = np.abs(gt-out)*4
      out = np.reshape(out, [1, out.shape[0], out.shape[1], out.shape[2]])
      batchviz = np.concatenate([inp, gt, out, diff], axis=0)
      batchviz = np.clip(batchviz, 0, 1)
      return batchviz

  def _get_psf_batch(self):
    for b in self.val_loader:
      psf = b[2].cpu().numpy()
      return psf / (np.max(psf) - np.min(psf))

  def on_epoch_begin(self, epoch):
    self.current_epoch = epoch
    #print(self.model.reg_kernels)
    #print(self.model.reg_kernel_weights)

  def on_epoch_end(self, epoch, logs):
    if "loss" in logs.keys():
      self.val_loss_viz.update(epoch, logs['loss'])
    if "psnr" in logs.keys():
      self.val_psnr_viz.update(epoch, logs['psnr'])

    im_batch = self._get_im_batch()
    self.viz.update(im_batch, per_row=self.val_loader.batch_size,
                    caption="input | gt | ours | ref | diff (x4)")
    psf_batch = self._get_psf_batch()
    self.psf_viz.update(psf_batch, per_row=self.val_loader.batch_size,
                        caption="PSF")

  def on_batch_end(self, batch, logs):
    frac = self.current_epoch + batch*1.0/self.num_batches
    if "loss" in logs.keys():
      self.loss_viz.update(frac, logs['loss'])
    if "psnr" in logs.keys():
      self.psnr_viz.update(frac, logs['psnr'])

def main(args):
  model = models.DeconvCG()

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  dset = datasets.DeconvDataset(args.dataset)
  val_dset = datasets.DeconvDataset(args.val_dataset)

  log.info("Training on {} with {} images".format(
    args.dataset, len(dset)))
  log.info("Validating on {} with {} images".format(
    args.val_dataset, len(val_dset)))

  if args.cuda:
    model = model.cuda()

  optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
  loss_fn = metrics.CroppedMSELoss(crop=5)
  psnr_fn = metrics.PSNR(crop=5)

  loader = DataLoader(dset, batch_size=1, num_workers=4, shuffle=True)
  val_loader = DataLoader(val_dset, batch_size=1)

  checkpointer = utils.Checkpointer(args.output, model, optimizer, verbose=False)
  callback = DeconvCallback(
      model, len(loader), val_loader, env="gapps_deconv")

  smooth_loss = 0
  smooth_psnr = 0
  #ema = 0.9
  ema = 0.0
  for epoch in range(args.num_epochs):
    # Training
    model.train(True)
    with tqdm(total=len(loader), unit=' batches') as pbar:
      pbar.set_description("Epoch {}/{}".format(epoch+1, args.num_epochs))
      callback.on_epoch_begin(epoch)
      for batch_id, batch in enumerate(loader):
        blurred, reference, kernel = batch
        blurred = Variable(blurred, requires_grad=False)
        reference = Variable(reference, requires_grad=False)
        kernel = Variable(kernel, requires_grad=False)

        if args.cuda:
          blurred = blurred.cuda()
          reference = reference.cuda()
          kernel = kernel.cuda()

        output = model(blurred, kernel)

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
        blurred, reference, kernel = batch
        blurred = Variable(blurred, requires_grad=False)
        reference = Variable(reference, requires_grad=False)
        kernel = Variable(kernel, requires_grad=False)

        if args.cuda:
          blurred = blurred.cuda()
          reference = reference.cuda()
          kernel = kernel.cuda()

        output = model(blurred, kernel)
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
  parser.add_argument("--dataset", default="images/filelist.txt")
  parser.add_argument("--val_dataset", default="images/filelist.txt")
  parser.add_argument("--output", default="output/deconv")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--num_epochs", type=int, default=10000)
  parser.add_argument("--cuda", type=bool, default=False)
  parser.add_argument("--batch_size", type=int, default=1)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_deconvolution')

  main(args)

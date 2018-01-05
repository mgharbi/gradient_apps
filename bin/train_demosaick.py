#!/usr/bin/env python
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
import gapps.demosaick as demosaick


log = logging.getLogger("gapps_demosaicking")


def main(args):
  model = models.LearnableDemosaick(
      num_filters=args.nfilters, fsize=args.fsize)
  # model.softmax_scale[...] = 0.01
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

  # log.info("Computing PCA filters")
  # vects = demosaick.get_pca_filters(dset, args.fsize)
  # model.sel_filts.data = th.from_numpy(vects)

  loader = DataLoader(dset, batch_size=args.batch_size, num_workers=4, shuffle=True)
  val_loader = DataLoader(val_dset, batch_size=args.batch_size)

  if args.cuda:
    model = model.cuda()
    # model.softmax_scale.cuda()

  # params = [p for n, p in model.named_parameters() if n != "green_filts"]
  optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

  mse_fn = metrics.CroppedMSELoss(crop=args.fsize//2)
  # l1_fn = metrics.CroppedL1Loss(crop=args.fsize//2)
  # grad_l1_fn = metrics.CroppedGradientLoss(crop=args.fsize//2)
  # loss_fn = lambda a,b : l1_fn(a, b) + grad_l1_fn(a, b)
  loss_fn = mse_fn
  psnr_fn = metrics.PSNR(crop=args.fsize//2)

  env = os.path.basename(args.output)
  checkpointer = utils.Checkpointer(args.output, model, optimizer, verbose=False)
  callback = demosaick.DemosaickCallback(
      model, reference_model, len(loader), val_loader, env=env)

  if args.chkpt is not None:
    log.info("Loading checkpoint {}".format(args.chkpt))
    checkpointer.load_checkpoint(args.chkpt)

  smooth_loss = 0
  smooth_psnr = 0
  ema = 0.9
  for epoch in range(args.num_epochs):
    callback.on_epoch_end(epoch, {})

    # Training
    model.train(True)
    with tqdm(total=len(loader), unit=' batches') as pbar:
      pbar.set_description("Epoch {}/{}".format(epoch+1, args.num_epochs))
      callback.on_epoch_begin(epoch)
      for batch_id, batch in enumerate(loader):
        mosaick, reference = batch
        mosaick = Variable(mosaick, requires_grad=False)
        reference = Variable(reference, requires_grad=False)

        if args.cuda:
          mosaick = mosaick.cuda()
          reference = reference.cuda()

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

        if args.cuda:
          mosaick = mosaick.cuda()
          reference = reference.cuda()

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
  parser.add_argument("--dataset", default="data/demosaick/train/filelist.txt")
  parser.add_argument("--val_dataset", default="data/demosaick/val/filelist.txt")
  parser.add_argument("--chkpt")
  parser.add_argument("--output", default="output/demosaick")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--num_epochs", type=int, default=3)
  parser.add_argument("--no-cuda", dest="cuda", action="store_false")
  parser.add_argument("--nfilters", type=int, default=9)
  parser.add_argument("--fsize", type=int, default=5)
  parser.set_defaults(cuda=True)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_demosaicking')

  main(args)

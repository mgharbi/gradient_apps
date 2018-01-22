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
  model = models.FancyDemosaick()
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
  val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True)

  if args.cuda:
    model = model.cuda()
    # model.softmax_scale.cuda()

  # params = [p for n, p in model.named_parameters() if n != "green_filts"]
  # optimizer = th.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
  optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

  # mse_fn = metrics.CroppedMSELoss(crop=args.fsize//2)
  l1_fn = th.nn.L1Loss()
  msssim_fn = metrics.MSSSIM()
  # grad_l1_fn = metrics.CroppedGradientLoss(crop=args.fsize//2)
  # loss_fn = lambda a, b: 0.84*msssim_fn(a, b) + (1-0.84)*l1_fn(a, b)
  alpha = args.alpha
  crop = 16 
  psnr_fn = metrics.PSNR(crop=args.fsize//2)

  env = os.path.basename(args.output)
  checkpointer = utils.Checkpointer(
      args.output, model, optimizer, verbose=False, interval=60)
  callback = demosaick.DemosaickCallback(
      model, reference_model, len(loader), val_loader, env=env)

  if args.regularize:
    log.info("Using L1 weight regularization")

  if args.chkpt is not None:
    log.info("Loading checkpoint {}".format(args.chkpt))
    checkpointer.load_checkpoint(args.chkpt, ignore_optim=True)
  else:
    chkpt_name, _ = checkpointer.load_latest()
    log.info("Resuming from latest checkpoint {}.".format(chkpt_name))

  ema = utils.ExponentialMovingAverage(["loss", "psnr", "psnr_g", "ssim", "l1"], alpha=0.99)
  for epoch in range(args.num_epochs):
    # callback.on_epoch_end(epoch, {})

    # Training
    model.train(True)
    with tqdm(total=len(loader), unit=' batches') as pbar:
      pbar.set_description("Epoch {}/{}".format(epoch+1, args.num_epochs))
      # callback.on_epoch_begin(epoch)
      callback.show_val_batch()
      for batch_id, batch in enumerate(loader):
        mosaick, reference = batch
        if args.green_only:
          reference[:, 0, ...] = 0.0
          reference[:, 2, ...] = 0.0
        mosaick = Variable(mosaick, requires_grad=False)
        reference = Variable(reference, requires_grad=False)

        if args.cuda:
          mosaick = mosaick.cuda()
          reference = reference.cuda()

        output = model(mosaick)

        # print output[:, 0].min(), output[:, 0].max(),
        # print output[:, 1].min(), output[:, 1].max(),
        # print output[:, 2].min(), output[:, 2].max()

        if args.green_only:
          output[:, 0, ...] = 0.0
          output[:, 2, ...] = 0.0

        optimizer.zero_grad()
        if crop > 0:
          output = output[:, :, crop:-crop, crop:-crop]
          reference = reference[:, :, crop:-crop, crop:-crop]

        ssim_ = 1-msssim_fn(output, reference)
        l1_ = l1_fn(output, reference)
        loss = ssim_*alpha + (1-alpha)*l1_
        if args.regularize:
          l1_reg = None
          reg_w = 1e-6
          for n, p in model.named_parameters():
            if l1_reg is None:
              l1_reg = p.norm(1)
            else:
              l1_reg = l1_reg + p.norm(1)
          loss += l1_reg*reg_w
        loss.backward()
        optimizer.step()


        psnr = psnr_fn(output, reference)

        psnr_green = psnr_fn(output[:, 1, ...], reference[:, 1, ...])

        ema.update("loss", loss.data[0]) 
        ema.update("psnr", psnr.data[0]) 
        ema.update("psnr_g", psnr_green.data[0]) 
        ema.update("ssim", ssim_.data[0]) 
        ema.update("l1", l1_.data[0]) 

        logs = {"loss": ema["loss"], "psnr": ema["psnr"], 
                "psnr_g": ema["psnr_g"],
                "ssim": ema["ssim"], "l1": ema["l1"]}
        pbar.set_postfix(logs)
        pbar.update(1)
        if pbar.n % args.viz_step == 0:
          callback.on_batch_end(batch_id, logs)
          callback.show_val_batch()
        checkpointer.periodic_checkpoint(epoch)

    # Validation
    model.train(False)
    with tqdm(total=len(val_loader), unit=' batches') as pbar:
      pbar.set_description("Epoch {}/{} (val)".format(epoch+1, args.num_epochs))
      avg = utils.Averager(["loss", "psnr", "ssim", "l1"])
      for batch_id, batch in enumerate(val_loader):
        mosaick, reference = batch
        mosaick = Variable(mosaick, requires_grad=False)
        reference = Variable(reference, requires_grad=False)

        if args.cuda:
          mosaick = mosaick.cuda()
          reference = reference.cuda()

        output = model(mosaick)
        if crop > 0:
          output = output[:, :, crop:-crop, crop:-crop]
          reference = reference[:, :, crop:-crop, crop:-crop]
        ssim_ = 1-msssim_fn(output, reference)
        l1_ = l1_fn(output, reference)
        loss = ssim_*alpha + (1-alpha)*l1_
        psnr = psnr_fn(output, reference)

        avg.update("loss", loss.data[0], count=mosaick.shape[0])
        avg.update("psnr", psnr.data[0], count=mosaick.shape[0])
        avg.update("ssim", ssim_.data[0], count=mosaick.shape[0])
        avg.update("l1", l1_.data[0], count=mosaick.shape[0])
        pbar.update(1)

      logs = {"loss": avg["loss"], "psnr": avg["psnr"],
              "ssim": avg["ssim"], "l1": avg["l1"]}

      pbar.set_postfix(logs)
      callback.on_epoch_end(epoch, logs)

    # save
    checkpointer.on_epoch_end(epoch)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="data/demosaick/train/filelist.txt")
  parser.add_argument("--val_dataset", default="data/demosaick/val/filelist.txt")
  parser.add_argument("--chkpt")
  parser.add_argument("--output", default="output/fancy_demosaick")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--num_epochs", type=int, default=100)
  parser.add_argument("--viz_step", type=int, default=10)
  parser.add_argument("--no-cuda", dest="cuda", action="store_false")
  parser.add_argument("--regularize", dest="regularize", action="store_true")
  parser.add_argument("--nfilters", type=int, default=9)
  parser.add_argument("--fsize", type=int, default=5)
  parser.add_argument("--alpha", type=float, default=0.84)
  parser.add_argument("--green_only", dest="green_only", action="store_true")
  parser.set_defaults(cuda=True, regularize=False, green_only=False)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_demosaicking')

  main(args)

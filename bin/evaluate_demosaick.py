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

import skimage.io

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
  reference_model = models.NaiveDemosaick()

  dset = datasets.DemosaickingDataset(
      args.dataset, transform=datasets.ToTensor())
  log.info("Validating on {} with {} images".format(
    args.dataset, len(dset)))

  loader = DataLoader(dset, batch_size=16, num_workers=4, shuffle=False)

  if args.cuda:
    model = model.cuda()

  l1_fn = metrics.CroppedL1Loss(crop=args.fsize//2)
  msssim_fn = metrics.MSSSIM()
  alpha = 0.84
  crop = args.fsize // 2
  psnr_fn = metrics.PSNR(crop=args.fsize//2)

  env = os.path.basename(args.chkpt)+"_eval"
  checkpointer = utils.Checkpointer(
      args.chkpt, model, None, verbose=False)
  chkpt_name, _ = checkpointer.load_latest()
  log.info("Loading checkpoint {}.".format(chkpt_name))

  callback = demosaick.DemosaickCallback(
      model, reference_model, len(loader), loader, env=env)

  idx = 0
  with tqdm(total=len(loader), unit=' batches') as pbar:
    pbar.set_description("Validation")
    avg = utils.Averager(["loss", "psnr", "ssim", "l1"])
    for batch_id, batch in enumerate(loader):
      mosaick, reference = batch
      mosaick = Variable(mosaick, requires_grad=False)
      reference = Variable(reference, requires_grad=False)

      if args.cuda:
        mosaick = mosaick.cuda()
        reference = reference.cuda()

      output = model(mosaick)

      if args.save is not None:
        if not os.path.exists(args.save):
          os.makedirs(args.save)
        for i in range(output.shape[0]):
          im = output[i].cpu().data.numpy()
          im = np.transpose(im, [1, 2, 0])
          im = np.clip(im, 0, 1)
          fname = os.path.join(args.save, "{:04d}.png".format(idx))
          idx += 1
          skimage.io.imsave(fname, im)

        

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
    callback.on_epoch_end(0, logs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--chkpt", default="output/demosaick")
  parser.add_argument("--save", default="output/demosaick_render")
  parser.add_argument("--dataset", default="data/demosaick/val/filelist.txt")
  parser.add_argument("--nfilters", type=int, default=9)
  parser.add_argument("--fsize", type=int, default=5)
  parser.add_argument("--no-cuda", dest="cuda", action="store_false")
  parser.set_defaults(cuda=True)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_demosaicking')

  main(args)

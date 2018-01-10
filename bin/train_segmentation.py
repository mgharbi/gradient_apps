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
import gapps.segmentation as seg
import gapps.metrics as metrics
import gapps.segmentation_models as models


log = logging.getLogger("gapps_segmentation")


def main(args):
  if not os.path.exists(args.output):
    os.makedirs(args.output)

  dset = datasets.ADESegmentationDataset(
      args.list_train, args)
  val_dset = datasets.ADESegmentationDataset(
      args.list_val, args)
  loader = DataLoader(dset, batch_size=args.batch_size, num_workers=4, shuffle=True)
  val_loader = DataLoader(val_dset, batch_size=args.batch_size)

  model = models.ReferenceSegmentation()
  crit = th.nn.NLLLoss2d(ignore_index=-1)

  if args.cuda:
    model = model.cuda()

  optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

  env = os.path.basename(args.output)
  checkpointer = utils.Checkpointer(
      args.output, model, optimizer, verbose=False, interval=600)
  callback = seg.Callback(
      model, len(loader), val_loader, env=env)

  chkpt_name, _ = checkpointer.load_latest()
  log.info("Resuming from latest checkpoint {}.".format(chkpt_name))

  ema = utils.ExponentialMovingAverage(["loss", "acc"])
  for epoch in range(args.num_epochs):
    # Training
    model.train(True)
    with tqdm(total=len(loader), unit=' batches') as pbar:
      # TODO: calculate accuracy
      pbar.set_description("Epoch {}/{}".format(epoch+1, args.num_epochs))
      callback.on_epoch_begin(epoch)
      for batch_id, batch in enumerate(loader):
        optimizer.zero_grad()
        pred, loss = seg.forward_with_loss(batch, model, crit, cuda=args.cuda)
        loss.backward()
        optimizer.step()

        acc, _ = metrics.accuracy(batch, pred)

        ema.update("loss", loss.data[0])
        ema.update("acc", acc) 

        logs = {"loss": ema["loss"], "acc": ema["acc"]*100}
        pbar.set_postfix(logs)
        pbar.update(1)
        checkpointer.periodic_checkpoint(epoch)

        # if pbar.n % 100 == 0:
        callback.on_batch_end(batch_id, logs)

        # if pbar.n == 5:
        #   break

    # Validation
    model.train(False)
    with tqdm(total=len(val_loader), unit=' batches') as pbar:
      pbar.set_description("Epoch {}/{} (val)".format(epoch+1, args.num_epochs))
      avg = utils.Averager(["loss", "acc"])
      for batch_id, batch in enumerate(val_loader):
        pred, loss = seg.forward_with_loss(batch, model, crit, cuda=args.cuda)

        acc, nvalid = metrics.accuracy(batch, pred)

        avg.update("loss", loss.data[0], count=pred.shape[0])
        avg.update("acc", acc, count=nvalid) 

        break

      logs = {"loss": avg["loss"], "acc": avg["acc"]*100}
      pbar.set_postfix(logs)

    callback.on_epoch_end(epoch, logs)

    # save
    checkpointer.on_epoch_end(epoch)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Path related arguments
  parser.add_argument('--list_train',
                      default='data/ade20k/ADE20K_object150_train.txt')
  parser.add_argument('--list_val',
                      default='data/ade20k/ADE20K_object150_val.txt')
  parser.add_argument('--root_img',
                      default='data/ade20k/ADEChallengeData2016/images')
  parser.add_argument('--root_seg',
                      default='data/ade20k/ADEChallengeData2016/annotations')

  # Data related arguments
  parser.add_argument('--num_val', default=128, type=int,
                      help='number of images to evalutate')
  parser.add_argument('--num_class', default=150, type=int,
                      help='number of classes')
  parser.add_argument('--imgSize', default=384, type=int,
                      help='input image size')
  parser.add_argument('--segSize', default=384, type=int,
                      help='output image size')

  parser.add_argument("--output", default="output/segmentation")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--num_epochs", type=int, default=100)
  parser.add_argument("--no-cuda", dest="cuda", action="store_false")
  parser.add_argument("--nfilters", type=int, default=9)
  parser.add_argument("--fsize", type=int, default=5)
  parser.set_defaults(cuda=True)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_segmentation')

  main(args)

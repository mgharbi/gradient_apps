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
import skimage.io
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchlib.viz as viz
import torchlib.utils as utils

import gapps.datasets as datasets
import gapps.modules as models
import gapps.metrics as metrics
import gapps

log = logging.getLogger("gapps_nlm")

class DenoiseCallback(object):
  def __init__(self, model, ref_model, val_loader, cuda, env=None):
    self.model = model
    self.ref_model = ref_model
    self.val_loader = val_loader
    self.cuda = cuda

    self.viz = viz.BatchVisualizer("denoise", port=args.port, env=env)

    self.loss_viz = viz.ScalarVisualizer("loss", port=args.port, env=env)
    self.psnr_viz = viz.ScalarVisualizer("psnr", port=args.port, env=env)
    self.val_loss_viz = viz.ScalarVisualizer("val_loss", port=args.port, env=env)
    self.val_psnr_viz = viz.ScalarVisualizer("val_psnr", port=args.port, env=env)
    self.ref_loss_viz = viz.ScalarVisualizer("ref_loss", port=args.port, env=env)
    self.ref_psnr_viz = viz.ScalarVisualizer("ref_psnr", port=args.port, env=env)

  def _get_im_batch(self):
    for b in self.val_loader:
      batchv = Variable(b[0])
      if args.cuda:
        batchv = batchv.cuda()

      out = self.model(batchv)
      out_ref = self.ref_model(batchv)
      out = out.data.cpu().numpy()
      out_ref = out_ref.data.cpu().numpy()

      inp = b[0].cpu().numpy()
      gt = b[1].cpu().numpy()
      diff = np.abs(gt-out)*4
      diff_ref = np.abs(gt-out_ref)*4
      batchviz = np.concatenate([inp, gt, out, out_ref, diff, diff_ref], axis=0)
      batchviz = np.clip(batchviz, 0, 1)
      return batchviz

  def on_validation_end(self, epoch, logs):
    if "val_loss" in logs.keys():
      self.val_loss_viz.update(epoch, logs['val_loss'])
    if "val_psnr" in logs.keys():
      self.val_psnr_viz.update(epoch, logs['val_psnr'])
    if "ref_loss" in logs.keys():
      self.ref_loss_viz.update(epoch, logs['ref_loss'])
    if "ref_psnr" in logs.keys():
      self.ref_psnr_viz.update(epoch, logs['ref_psnr'])

    im_batch = self._get_im_batch()
    self.viz.update(im_batch, per_row=self.val_loader.batch_size,
                    caption="input | gt | ours (train) | ref | diff (x4)")

  def on_iteration_end(self, iteration, logs):
    if "loss" in logs.keys():
      self.loss_viz.update(iteration, logs['loss'])
    if "psnr" in logs.keys():
      self.psnr_viz.update(iteration, logs['psnr'])

def main(args):
  model = models.NonLocalMeans()
  ref_model = models.NonLocalMeans()

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  dset = datasets.DenoiseDataset(args.dataset)
  val_dset = datasets.DenoiseDataset(args.val_dataset, is_validate = True)

  log.info("Training on {} with {} images".format(
    args.dataset, len(dset)))
  log.info("Validating on {} with {} images".format(
    args.val_dataset, len(val_dset)))

  if args.cuda:
    model = model.cuda()
    ref_model = ref_model.cuda()
  print("Training parameters:")
  params_to_train = []
  for n, p in model.named_parameters():
    print("  -", n)
    params_to_train.append(p)
  optimizer = th.optim.Adam(params_to_train, lr=args.lr)
  # optimizer = th.optim.SGD(model.parameters(), lr=args.lr)
  loss_fn = metrics.CroppedL1Loss(crop=5)
  psnr_fn = metrics.PSNR(crop=5)

  loader = DataLoader(dset, batch_size=args.batch_size, num_workers=1, shuffle=True)
  val_loader = DataLoader(val_dset, batch_size=8)

  checkpointer = utils.Checkpointer(args.output, model, optimizer, verbose=True)
  callback = DenoiseCallback(
      model, ref_model, val_loader, args.cuda, env="gapps_nlm")

  smooth_loss = 0
  smooth_psnr = 0
  ema = 0.9
  chkpt_name, iteration = checkpointer.load_latest()
  log.info("Resuming from latest checkpoint {}.".format(chkpt_name))
  train_iterator = iter(loader)
  best_psnr = 0.0
  first = True
  while True:
    # Training
    # Get a batch from the dataset
    try:
        batch = train_iterator.next()
    except StopIteration:
        train_iterator = iter(loader)
        batch = train_iterator.next()
    model.train(True)

    # Setup input & reference
    noisy, reference = batch
    noisy = Variable(noisy, requires_grad=False)
    reference = Variable(reference, requires_grad=False)

    # Transfer data to gpu if necessary
    if args.cuda:
      noisy = noisy.cuda()
      reference = reference.cuda()

    # Run the model
    output = model(noisy)

    # Compute loss & optimize
    optimizer.zero_grad()
    loss = loss_fn(output, reference)
    loss.backward()
    optimizer.step()

    # Compute PSNR
    psnr = psnr_fn(output, reference)

    # Exponential smooth of error curve
    if first:
      smooth_loss = loss.data[0]
      smooth_psnr = psnr.data[0]
      first = False
    else:
      smooth_loss = ema*smooth_loss + (1-ema)*loss.data[0]
      smooth_psnr = ema*smooth_psnr + (1-ema)*psnr.data[0]

    print('loss: {}, psnr: {}'.format(smooth_loss, smooth_psnr))
    model.train(False)
    ref_model.train(False)

    logs = {"loss": smooth_loss, "psnr": smooth_psnr}
    callback.on_iteration_end(iteration, logs)

    if iteration % 20 == 0:
      # Validation
      # Go through the whole validation dataset
      total_loss = 0
      total_psnr = 0
      total_ref_loss = 0
      total_ref_psnr = 0
      n_seen = 0
      for batch_id, batch in enumerate(val_loader):
        noisy, reference = batch
        noisy = Variable(noisy, requires_grad=False)
        reference = Variable(reference, requires_grad=False)

        if args.cuda:
          noisy = noisy.cuda()
          reference = reference.cuda()

        output = model(noisy)
        loss = loss_fn(output, reference)
        psnr = psnr_fn(output, reference)

        ref_output = ref_model(noisy)
        ref_loss = loss_fn(ref_output, reference)
        ref_psnr = psnr_fn(ref_output, reference)

        total_loss += loss.data[0]*args.batch_size
        total_psnr += psnr.data[0]*args.batch_size
        total_ref_loss += ref_loss.data[0]*args.batch_size
        total_ref_psnr += ref_psnr.data[0]*args.batch_size
        n_seen += args.batch_size

      val_loss = total_loss / n_seen
      val_psnr = total_psnr / n_seen
      ref_loss = total_ref_loss / n_seen
      ref_psnr = total_ref_psnr / n_seen

      logs = {"val_loss": val_loss, "val_psnr": val_psnr,
              "ref_loss": ref_loss, "ref_psnr": ref_psnr}

      callback.on_validation_end(iteration, logs)
      # save
      checkpointer.on_epoch_end(iteration)
      # save best
      if val_psnr > best_psnr:
        filename = 'epoch_{:03d}_best.pth.tar'.format(iteration+1)
        checkpointer.save_checkpoint(iteration, filename)
        best_psnr = val_psnr

    iteration += 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  #parser.add_argument("--dataset", default="images/filelist.txt")
  #parser.add_argument("--val_dataset", default="images/filelist.txt")
  parser.add_argument("--dataset", default="/data/graphics/approximation/datasets/imagenet/imagenet_jpeg.txt")
  parser.add_argument("--val_dataset", default="/data/graphics/approximation/datasets/imagenet/imagenet_jpeg.txt")
  parser.add_argument("--output", default="output/denoise")
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--cuda", type=bool, default=False)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--port", type=int, default=8888)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_denoise')

  main(args)


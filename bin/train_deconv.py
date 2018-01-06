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

log = logging.getLogger("gapps_deconvolution2")

irls_iter = 1
cg_iter = 20
ref_irls_iter = 1
ref_cg_iter = 20

class DeconvCallback(object):
  def __init__(self, model, ref_model, val_loader, cuda, env=None):
    self.model = model
    self.ref_model = ref_model
    self.val_loader = val_loader
    self.cuda = cuda

    self.viz = viz.BatchVisualizer("deconv", port=args.port, env=env)
    self.psf_viz = viz.BatchVisualizer("psf", port=args.port, env=env)
    self.data_kernels0_viz = viz.BatchVisualizer("data_kernels0", port=args.port, env=env)
    self.data_kernels1_viz = viz.BatchVisualizer("data_kernels1", port=args.port, env=env)
    self.data_kernel_weights0_viz = viz.ScalarVisualizer("data_kernel_weights0",
      ntraces = self.model.data_kernel_weights.shape[1], port=args.port, env=env)
    self.data_kernel_weights1_viz = viz.ScalarVisualizer("data_kernel_weights1",
      ntraces = self.model.data_kernel_weights.shape[1], port=args.port, env=env)
    self.reg_kernels0_viz = viz.BatchVisualizer("reg_kernels0", port=args.port, env=env)
    self.reg_kernels1_viz = viz.BatchVisualizer("reg_kernels1", port=args.port, env=env)
    self.reg_kernel_weights0_viz = viz.ScalarVisualizer("reg_kernel_weights0",
      ntraces = self.model.reg_kernel_weights.shape[1], port=args.port, env=env)
    self.reg_kernel_weights1_viz = viz.ScalarVisualizer("reg_kernel_weights1",
      ntraces = self.model.reg_kernel_weights.shape[1], port=args.port, env=env)
    self.filter_s_viz = viz.ScalarVisualizer("filter_s0",
      ntraces = self.model.filter_s.shape[1], port=args.port, env=env)
    self.filter_r_viz = viz.ScalarVisualizer("filter_r0",
      ntraces = self.model.filter_r.shape[1], port=args.port, env=env)
    self.reg_thresholds_viz = viz.ScalarVisualizer("reg_thresholds",
      ntraces = self.model.reg_thresholds.shape[1], port=args.port, env=env)

    self.loss_viz = viz.ScalarVisualizer("loss", port=args.port, env=env)
    self.psnr_viz = viz.ScalarVisualizer("psnr", port=args.port, env=env)
    self.val_loss_viz = viz.ScalarVisualizer("val_loss", port=args.port, env=env)
    self.val_psnr_viz = viz.ScalarVisualizer("val_psnr", port=args.port, env=env)
    self.ref_loss_viz = viz.ScalarVisualizer("ref_loss", port=args.port, env=env)
    self.ref_psnr_viz = viz.ScalarVisualizer("ref_psnr", port=args.port, env=env)

  def _get_im_batch(self):
    for b in self.val_loader:
      batchv = Variable(b[0])
      psf = Variable(b[2])
      if args.cuda:
        batchv = batchv.cuda()
        psf = psf.cuda()

      out = self.model(batchv, psf, ref_irls_iter, ref_cg_iter)
      out_ref = self.ref_model(batchv, psf, ref_irls_iter, ref_cg_iter)
      out = out.data.cpu().numpy()
      out_ref = out_ref.data.cpu().numpy()

      inp = b[0].cpu().numpy()
      gt = b[1].cpu().numpy()
      diff = np.abs(gt-out)*4
      diff_ref = np.abs(gt-out_ref)*4
      batchviz = np.concatenate([inp, gt, out, out_ref, diff, diff_ref], axis=0)
      batchviz = np.clip(batchviz, 0, 1)
      return batchviz

  def _get_psf_batch(self):
    for b in self.val_loader:
      psf = b[2].cpu().numpy()

      psf = psf / (np.max(psf) - np.min(psf))
      psf = np.reshape(psf, [psf.shape[0], 1, psf.shape[1], psf.shape[2]])
      return psf

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
    psf_batch = self._get_psf_batch()
    self.psf_viz.update(psf_batch, per_row=self.val_loader.batch_size,
                        caption="PSF")

  def on_iteration_end(self, iteration, logs):
    if "loss" in logs.keys():
      self.loss_viz.update(iteration, logs['loss'])
    if "psnr" in logs.keys():
      self.psnr_viz.update(iteration, logs['psnr'])

    dk0 = self.model.data_kernels.data[0, :, :, :].cpu().numpy()
    dk0 = np.reshape(dk0, [dk0.shape[0], 1, dk0.shape[1], dk0.shape[2]])
    dk0 = (dk0 / np.abs(dk0).max() + 1.0) / 2.0
    self.data_kernels0_viz.update(dk0,
                                 per_row=dk0.shape[0],
                                 caption="data_kernels0")
    dk1 = self.model.data_kernels.data[1, :, :, :].cpu().numpy()
    dk1 = np.reshape(dk1, [dk1.shape[0], 1, dk1.shape[1], dk1.shape[2]])
    dk1 = (dk1 / np.abs(dk1).max() + 1.0) / 2.0
    self.data_kernels0_viz.update(dk1,
                                 per_row=dk1.shape[0],
                                 caption="data_kernels1")
    self.data_kernel_weights0_viz.update(iteration, self.model.data_kernel_weights.data[0, :].cpu().numpy())
    self.data_kernel_weights1_viz.update(iteration, self.model.data_kernel_weights.data[1, :].cpu().numpy())

    rk0 = self.model.reg_kernels.data[0, :, :, :].cpu().numpy()
    rk0 = np.reshape(rk0, [rk0.shape[0], 1, rk0.shape[1], rk0.shape[2]])
    rk0 = (rk0 / np.abs(rk0).max() + 1.0) / 2.0
    self.reg_kernels0_viz.update(rk0,
                                 per_row=rk0.shape[0],
                                 caption="reg_kernels0")
    rk1 = self.model.reg_kernels.data[1, :, :, :].cpu().numpy()
    rk1 = np.reshape(rk1, [rk1.shape[0], 1, rk1.shape[1], rk1.shape[2]])
    rk1 = (rk1 / np.abs(rk1).max() + 1.0) / 2.0
    self.reg_kernels1_viz.update(rk1,
                                 per_row=rk1.shape[0],
                                 caption="reg_kernels1")
    self.reg_kernel_weights0_viz.update(iteration, self.model.reg_kernel_weights.data[0, :].cpu().numpy())
    self.reg_kernel_weights1_viz.update(iteration, self.model.reg_kernel_weights.data[1, :].cpu().numpy())
    self.filter_s_viz.update(iteration, self.model.filter_s.data[0, :].cpu().numpy())
    self.filter_r_viz.update(iteration, self.model.filter_r.data[0, :].cpu().numpy())
    self.reg_thresholds_viz.update(iteration, self.model.reg_thresholds.data[0, :].cpu().numpy())

def main(args):
  model = models.DeconvCG(num_stages = 3)
  ref_model = models.DeconvCG(ref = True)

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  dset = datasets.DeconvDataset(args.dataset)
  val_dset = datasets.DeconvDataset(args.val_dataset, is_validate = True)

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
  loss_fn = metrics.CroppedL1Loss(crop=20)
  psnr_fn = metrics.PSNR(crop=20)

  loader = DataLoader(dset, batch_size=args.batch_size, num_workers=1, shuffle=True)
  val_loader = DataLoader(val_dset, batch_size=8)

  checkpointer = utils.Checkpointer(args.output, model, optimizer, verbose=True)
  callback = DeconvCallback(
      model, ref_model, val_loader, args.cuda, env="gapps_deconv2")

  smooth_loss = 0
  smooth_psnr = 0
  ema = 0.9
  chkpt_name, iteration = checkpointer.load_latest()
  log.info("Resuming from latest checkpoint {}.".format(chkpt_name))
  train_iterator = iter(loader)
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
    blurred, reference, kernel = batch
    blurred = Variable(blurred, requires_grad=False)
    reference = Variable(reference, requires_grad=False)
    kernel = Variable(kernel, requires_grad=False)

    # Transfer data to gpu if necessary
    if args.cuda:
      blurred = blurred.cuda()
      reference = reference.cuda()
      kernel = kernel.cuda()

    # Run the model
    output = model(blurred, kernel, irls_iter, cg_iter)

    # Compute loss & optimize
    optimizer.zero_grad()
    loss = loss_fn(output, reference)
    loss.backward()
    for n, p in model.named_parameters():
        print('name:', n)
        print('val', p)
        print('grad:', p.grad)
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
        blurred, reference, kernel = batch
        blurred = Variable(blurred, requires_grad=False)
        reference = Variable(reference, requires_grad=False)
        kernel = Variable(kernel, requires_grad=False)

        if args.cuda:
          blurred = blurred.cuda()
          reference = reference.cuda()
          kernel = kernel.cuda()

        output = model(blurred, kernel, ref_irls_iter, ref_cg_iter)
        loss = loss_fn(output, reference)
        psnr = psnr_fn(output, reference)

        ref_output = ref_model(blurred, kernel, ref_irls_iter, ref_cg_iter)
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

    iteration += 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  #parser.add_argument("--dataset", default="images/filelist.txt")
  #parser.add_argument("--val_dataset", default="images/filelist.txt")
  parser.add_argument("--dataset", default="/data/graphics/approximation/datasets/imagenet/imagenet_jpeg.txt")
  parser.add_argument("--val_dataset", default="/data/graphics/approximation/datasets/imagenet/imagenet_jpeg.txt")
  parser.add_argument("--output", default="output/deconv")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--cuda", type=bool, default=True)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--port", type=int, default=8888)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_deconvolution2')

  main(args)

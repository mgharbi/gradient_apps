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

from torchvision import transforms

import gapps.dataset as dset
import gapps.modules.models as models
# import gapps.viz as viz

log = logging.getLogger("gapps")

def save(checkpoint, model, optimizer, step):
  log.info("saving checkpoint {} at step {}".format(checkpoint, step))
  th.save({
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'step': step,
    } , checkpoint)

def make_variable(d, cuda=True):
  ret = {}
  for k in d.keys():
    if cuda:
      ret[k] = Variable(d[k].cuda())
    else:
      ret[k] = Variable(d[k])
  return ret

def crop_like(src, tgt):
  src_sz = np.array(src.shape)
  tgt_sz = np.array(tgt.shape)
  crop = (src_sz-tgt_sz)[2:4] // 2
  if (crop > 0).any():
    return src[:, :, crop[0]:src_sz[2]-crop[0], crop[1]:src_sz[3]-crop[1]]
  else:
    return src

def main(args):
  data = dset.ADESegmentationDataset(
      args.data_dir,
      transform=dset.ToTensor())

  dataloader = DataLoader(data, 
      batch_size=args.batch_size,
      shuffle=True, num_workers=4)

  log.info("Training with {} samples".format(len(dataloader)))

  # model = models.MotionSamples()
  #
  # loss_fn = th.nn.MSELoss()
  # optimizer = optim.Adam(model.parameters(), lr=args.lr)

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  checkpoint = os.path.join(args.output, "checkpoint.ph")
  if args.checkpoint is not None:
    checkpoint = args.checkpoint

  global_step = 0

  if os.path.isfile(checkpoint):
    log.info("Resuming from checkpoint {}".format(checkpoint))
    chkpt = th.load(checkpoint)
    model.load_state_dict(chkpt['state_dict'])
    optimizer.load_state_dict(chkpt['optimizer'])
    global_step = chkpt['step']

  checkpoint = os.path.join(args.output, "checkpoint.ph")

  # loss_viz = viz.ScalarVisualizer("loss")
  # image_viz = viz.BatchVisualizer("images")
  #
  # print(model)
  #
  # model.cuda()
  # loss_fn.cuda()
  #
  log.info("Starting training from step {}".format(global_step))

  smooth_loss = 0
  smooth_time = 0
  ema_alpha = 0.5
  last_checkpoint_time = time.time()
  try:
    # for epoch in range(args.epochs):
    epoch = 0
    while True:
      log.info("Starting epoch {}".format(epoch+1))
      # Train for one epoch
      for step, batch in enumerate(dataloader):
        batch_start = time.time()
        frac_epoch =  epoch+1.0*step/len(dataloader)
        optimizer.zero_grad()

        batch_v = make_variable(batch, cuda=True)
        # output = model(batch_v)
        # target = crop_like(batch_v['target_image'], output)
        # loss = loss_fn(output, target)

        # loss.backward()
        optimizer.step()
        global_step += 1

        batch_end = time.time()
        smooth_loss = ema_alpha*loss.data[0] + (1.0-ema_alpha)*smooth_loss
        smooth_time = ema_alpha*(batch_end-batch_start) + (1.0-ema_alpha)*smooth_time

        if global_step % args.log_step == 0:
          log.info("Epoch {:.1f} | loss = {:.4f} | {:.1f} samples/s".format(frac_epoch, smooth_loss, target.shape[0]/smooth_time))

        if global_step % args.viz_step == 0:
          log.info("viz at step {}".format(global_step))

          for val_batch in val_dataloader:
            val_batchv = make_variable(val_batch, cuda=True)
            output = model(val_batchv)
            target = crop_like(val_batchv['target_image'], output)
            lowspp = crop_like(val_batchv['low_spp'], output)
            imgs = np.clip(th.cat((lowspp, target, output), 0).cpu().data, 0, 1)

            # image_viz.update(imgs,
            #     caption="Epoch {:.1f} | {}spp, target, output".format(frac_epoch, batch["0_radiance_direct"].shape[-1]),
            #                 per_row=lowspp.shape[0])
            break  # Only one batch for validation

          # loss_viz.update(frac_epoch, smooth_loss)

        if batch_end-last_checkpoint_time > args.checkpoint_interval:
          last_checkpoint_time = time.time()
          save(checkpoint, model, optimizer, global_step)


      epoch += 1
      if args.epochs > 0 and epoch >= args.epochs:
        log.info("Ending training at epoch {} of {}".format(epoch, args.epochs))
        break
      # TODO: Save model at end of epoch

  except KeyboardInterrupt:
    log.info("training interrupted at step {}".format(global_step))

  save(checkpoint, model, optimizer, global_step)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir')
  parser.add_argument('output')
  parser.add_argument('--checkpoint')
  parser.add_argument('--epochs', type=int, default=-1)
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--debug', dest="debug", action="store_true")

  parser.add_argument('--log_step', type=int, default=100)
  parser.add_argument('--checkpoint_interval', type=int, default=600, help='in seconds')
  parser.add_argument('--viz_step', type=int, default=1000)
  parser.set_defaults(debug=False)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  if args.debug:
    log.setLevel(logging.DEBUG)
  else:
    log.setLevel(logging.INFO)
  setproctitle.setproctitle(
      'gapps_{}'.format(os.path.basename(args.output)))

  main(args)

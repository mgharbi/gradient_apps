import argparse
import os
import time
import unittest
import logging
import setproctitle
import skimage.io
import skimage.transform as xform

import cv2
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler
import tifffile as tiff

import torchlib.utils as utils
import gapps.functions as funcs
import gapps.modules as modules
import gapps.utils as gutils

log = logging.getLogger("gapps_burst_demosaicking")

def load_images(dataset, crop_x, crop_y, tile_size, max_images):
  images = []
  for f in os.listdir(dataset):
    ext = os.path.splitext(f)[-1]
    if ext == ".tiff":
      im = tiff.imread(os.path.join(dataset, f))
      im = im.astype(np.float32)/(1.0*2**16)
      print im[..., 0].max(), im[..., 1].max(), im[..., 2].max()
    elif ext == ".png":
      im = skimage.io.imread(os.path.join(dataset, f))
      im = im[..., 0].astype(np.float32)/255.0
    else:
      continue
    print f
    im = im[crop_y:crop_y+tile_size, crop_x:crop_x+tile_size]
    im = np.expand_dims(im, 0)
    im = th.from_numpy(im)
    images.append(im)
    if max_images is not None and len(images) == max_images:
      break
  images = th.cat(images, 0)
  return images

def load_synth(dataset, crop_x, crop_y, tile_size, max_images):
  images = []
  for f in os.listdir(dataset):
    ext = os.path.splitext(f)[-1]
    if ext == ".tiff":
      image = tiff.imread(os.path.join(dataset, f))
      image = image.astype(np.float32)/(1.0*2**16)
      image = image[::2, ::2, :]  # subsample

      h, w, _ = image.shape

      start_y = h//2-tile_size//2
      start_x = w//2-tile_size//2

      for count in range(max_images):
        im = xform.rotate(image, np.random.uniform(-30, 30)).astype(np.float32)

        im = im[start_y:start_y+tile_size, start_x:start_x+tile_size, :]

        im = gutils.make_mosaick(im)
        im = np.copy(np.expand_dims(im, 0))
        im = th.from_numpy(im)
        images.append(im)
      break
  images = th.cat(images, 0)
  return images

def init_homographies(images):
  FLANN_INDEX_KDTREE = 0
  MIN_MATCH_COUNT = 10

  log.info("Extracting SIFT")
  images = (images.cpu().numpy()*255).astype(np.uint8)
  sift = cv2.xfeatures2d.SIFT_create()
  ref = images[0]

  kp_ref, des_ref = sift.detectAndCompute(ref, None)

  log.info("Matching features")
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)

  homographies = np.zeros((images.shape[0], 8))

  log.info("Computing alignments")
  for i in range(0, images.shape[0]):
    log.info(" {} of {}".format(i+1, images.shape[0]))
    kp, des = sift.detectAndCompute(images[i], None)
    matches = flann.knnMatch(des, des_ref, k=2)
    good = []
    for m, n in matches:
      if m.distance < 0.7*n.distance:
        good.append(m)

    if len(good) > MIN_MATCH_COUNT:
      src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp_ref[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
      # matchesMask = mask.ravel().tolist()
      # M = np.linalg.inv(M)

      homographies[i, :] = np.ravel(M)[:-1]

      # h, w = ref.shape
      # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
      # dst = cv2.perspectiveTransform(pts,M)
      # img = cv2.polylines(images[i],[np.int32(dst)],True,255,3, cv2.LINE_AA)
      #
      # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
      #                    singlePointColor = None,
      #                    matchesMask = matchesMask, # draw only inliers
      #                    flags = 2)
      #
      # img3 = cv2.drawMatches(ref,kp_ref,img,kp,good,None,**draw_params)
      # from matplotlib import pyplot as plt
      # plt.imshow(img3, 'gray'),plt.show()
    
    else:
      print "not enough MATCHES"

  h, w = images.shape[1:]
  homographies[:, 2] /= w*1.0
  homographies[:, 5] /= h*1.0
  homographies = th.from_numpy(homographies.astype(np.float32))
  return homographies

def init_reconstruction(images, init_idx=0, scale=1):
  # init to first image, with naive demosaick
  n, h, w = images.shape
  im = images[init_idx].view(1, 1, h, w)
  demosaicking = modules.NaiveDemosaick()
  init = th.clamp(demosaicking(im).squeeze(0), 0, 1)
  if scale != 1:
    init = np.transpose(init.numpy(), [1, 2, 0])
    init = xform.rescale(init, scale).astype(np.float32)
    init = np.transpose(init, [2, 0, 1])
    return th.from_numpy(init)
  else:
    return init.data.clone()

def main(args):
  output = os.path.join(args.output, os.path.basename(args.dataset))
  if not os.path.exists(output):
    os.makedirs(output)

  log.info("Loading inputs")
  if args.synth:
    images = load_synth(args.dataset, args.x, args.y, args.tile_size, args.max_images)
  else:
    images = load_images(args.dataset, args.x, args.y, args.tile_size, args.max_images)
  n, h, w = images.shape


  log.info("Image dimensions {}x{}x{}".format(n, h, w))

  log.info("Initializing homographies")
  homographies = init_homographies(images)

  log.info("Initializing reconstruction")
  recons = init_reconstruction(images, scale=args.scale)

  log.info("Reconstruction at {}x{}x{}".format(*recons.shape))

  init = recons.clone()
  identity_transform = homographies.clone()
  identity_transform.zero_()
  identity_transform[:, 0] = 1.0
  identity_transform[:, 4] = 1.0
  gradient_weight = th.ones(1)*args.gradient_weight

  op = modules.BurstDemosaicking()

  if args.cuda:
    log.info("Running cuda")
    op = op.cuda()
    recons = recons.cuda()
    homographies = homographies.cuda()
    images = images.cuda()
    gradient_weight = gradient_weight.cuda()
    identity_transform = identity_transform.cuda()

  images = Variable(images, False)
  homographies = Variable(homographies, True)
  recons = Variable(recons, True)
  identity_transform = Variable(identity_transform, True)
  gradient_weight = Variable(gradient_weight, False)

  if args.use_lbfgs:
    optimizer = th.optim.LBFGS([homographies, recons], lr=args.lr, max_iter=20, history_size=10)
  else:
    pre_optimizer = th.optim.Adam(
        [{'params': [homographies], 'lr': args.homography_lr}])
    optimizer = th.optim.Adam(
        [{'params': [recons], 'lr': args.lr},
          {'params': [homographies], 'lr': args.homography_lr}])


  log.info("Optimizing")
  ema = utils.ExponentialMovingAverage(["loss"], alpha=0.9)
  for step in range(args.presteps+args.steps):
    if args.use_lbfgs:
      raise NotImplemented
      def closure():
        optimizer.zero_grad()
        loss, reproj = op(images, homographies, recons, gradient_weight)
        loss.backward()
        return loss
      loss = optimizer.step(closure)
    else:
      if step < args.presteps:
        pre_optimizer.zero_grad()
      else:
        optimizer.zero_grad()
      loss, reproj = op(images, homographies, recons, gradient_weight)
      loss.backward()
      if step < args.presteps:
        step_label = "PreStep"
        pre_optimizer.step()
      else:
        step_label = "Step"
        optimizer.step()
    # regul = th.pow(homographies-identity_transform, 2).mean()
    # print loss, regul
    # loss = loss + args.regularization*regul

    ema.update("loss", loss.data[0]) 
    log.info("{} {} loss = {:.6f}".format(step_label, step, ema["loss"]))


  log.info("Producing outputs")

  for i in range(n):
    log.info(("{:.2f} "*8).format(*list(homographies.cpu().data[i].numpy())))

  diff = (init - recons.data.cpu()).abs().numpy()
  log.info("Max diff {}".format(diff.max()))
  diff = np.squeeze(diff)*10
  diff = np.clip(np.transpose(diff, [1, 2, 0]), 0, 1)
  skimage.io.imsave(
      os.path.join(output, "diff_from_init.png"), diff)

  demosaicking = modules.NaiveDemosaick()
  for i in range(n):
    im = images[i].data.view(1, 1, images.shape[1], images.shape[2])
    out = demosaicking(im.cpu())
    out = out.data[0].cpu().numpy()
    out = np.clip(np.transpose(out, [1, 2, 0]), 0, 1)
    out = np.squeeze(out)
    skimage.io.imsave(
        os.path.join(output, "input_demosaicked_{:02d}.png".format(i)), out)

  for i in range(n):
    im = reproj[i].data.view(1, 1, reproj.shape[1], reproj.shape[2])
    out = im.abs()*10
    out = out[0].cpu().numpy()
    out = np.clip(np.transpose(out, [1, 2, 0]), 0, 1)
    out = np.squeeze(out)
    skimage.io.imsave(
        os.path.join(output, "reproj_{:02d}.png".format(i)), out)


  out = recons.data.cpu().numpy()
  out = np.clip(np.transpose(out, [1, 2, 0]), 0, 1)
  out = np.squeeze(out)
  skimage.io.imsave(
      os.path.join(output, "reconstruction.png"), out)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", default="output/burst_demosaick")
  parser.add_argument("--dataset", default="data/burst_demosaick/hydrant")
  parser.add_argument("--x", default=0, type=int)
  parser.add_argument("--y", default=0, type=int)
  parser.add_argument("--scale", default=1, type=float)
  parser.add_argument("--max_images", type=int)
  parser.add_argument("--tile_size", default=512, type=int)
  parser.add_argument("--presteps", default=10, type=int)
  parser.add_argument("--steps", default=10, type=int)
  parser.add_argument("--gradient_weight", default=1e-1, type=float)
  parser.add_argument("--lr", default=1e-4, type=float)
  parser.add_argument("--homography_lr", default=1e-4, type=float)
  parser.add_argument("--var_factor", default=1e2, type=float)
  parser.add_argument("--regularization", default=1e-3, type=float)
  parser.add_argument("--cuda", action='store_true', dest="cuda")
  parser.add_argument("--synth", action='store_true', dest="synth")
  parser.add_argument("--use_lbfgs", action='store_true', dest="use_lbfgs")
  parser.set_defaults(cuda=False, use_lbfgs=False, synth=False)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_demosaicking')

  main(args)

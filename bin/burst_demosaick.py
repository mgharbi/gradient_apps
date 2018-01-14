import argparse
import os
import time
import unittest
import logging
import setproctitle
import skimage.io

import cv2
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler
import tifffile as tiff

import gapps.utils as utils
import gapps.functions as funcs
import gapps.modules as modules

log = logging.getLogger("gapps_burst_demosaicking")

def load_images(dataset, crop_x, crop_y, tile_size):
  images = []
  for f in os.listdir(dataset):
    print f
    if os.path.splitext(f)[-1] == ".tiff":
      im = tiff.imread(os.path.join(dataset, f))
      im = im.astype(np.float32)/(1.0*2**16)
      im = im[crop_y:crop_y+tile_size, crop_x:crop_x+tile_size]
      im = np.expand_dims(im, 0)
      im = th.from_numpy(im)
      images.append(im)
  images = th.cat(images, 0)
  return images

def init_homographies(images):
  FLANN_INDEX_KDTREE = 0
  MIN_MATCH_COUNT = 10

  images = (images.cpu().numpy()*255).astype(np.uint8)
  sift = cv2.xfeatures2d.SIFT_create()
  ref = images[0]

  kp_ref, des_ref = sift.detectAndCompute(ref, None)

  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)

  homographies = np.zeros((images.shape[0], 8))

  for i in range(0, images.shape[0]):
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

  homographies = th.from_numpy(homographies.astype(np.float32))
  return homographies

def init_reconstruction(images, init_idx=0):
  # init to first image, with naive demosaick
  n, h, w = images.shape
  im = images[init_idx].view(1, 1, h, w)
  demosaicking = modules.NaiveDemosaick()
  init = demosaicking(im)
  return init.data.clone().squeeze(0)

def main(args):
  output = os.path.join(args.output, os.path.basename(args.dataset))
  if not os.path.exists(output):
    os.makedirs(output)

  images = load_images(args.dataset, args.x, args.y, args.tile_size)
  n, h, w = images.shape

  log.info("Image dimensions {}x{}x{}".format(n, h, w))

  homographies = init_homographies(images)

  recons = init_reconstruction(images)
  init = recons.clone()
  identity_transform = homographies.clone()
  identity_transform.zero_()
  identity_transform[:, 0] = 1.0
  identity_transform[:, 4] = 1.0
  gradient_weight = th.ones(1)*args.gradient_weight

  op = modules.BurstDemosaicking()


  if args.cuda:
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

  optimizer = th.optim.SGD(
      [{'params': [homographies], 'lr': args.homographies_lr},
        {'params': [recons], 'lr': args.recons_lr}],
      momentum=0.9, nesterov=True)

  for step in range(args.steps):
    optimizer.zero_grad()
    loss, reproj = op(images, homographies, recons, gradient_weight)
    regul = th.pow(homographies-identity_transform, 2).mean()
    # print loss, regul
    loss = loss + args.regularization*regul
    # loss.backward()
    # optimizer.step()
    log.info("Step {} loss = {:.4f}".format(step, loss.data[0]))

  for i in range(n):
    log.info(("{:.2f} "*8).format(*list(homographies.cpu().data[i].numpy())))

  demosaicking = modules.NaiveDemosaick()
  for i in range(n):
    im = images[i].data.view(1, 1, images.shape[1], images.shape[2])
    out = demosaicking(im)
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

  diff = (init - recons.data.cpu()).abs().numpy()
  print "Max diff", diff.max()
  diff = np.squeeze(diff)*100
  diff = np.clip(np.transpose(diff, [1, 2, 0]), 0, 1)
  skimage.io.imsave(
      os.path.join(output, "diff_from_init.png"), diff)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", default="output/burst_demosaick")
  parser.add_argument("--dataset", default="data/burst_demosaick/hydrant")
  parser.add_argument("--x", default=0, type=int)
  parser.add_argument("--y", default=0, type=int)
  parser.add_argument("--tile_size", default=512, type=int)
  parser.add_argument("--steps", default=10, type=int)
  parser.add_argument("--gradient_weight", default=1e-1, type=float)
  parser.add_argument("--recons_lr", default=1e0, type=float)
  parser.add_argument("--homographies_lr", default=1e-6, type=float)
  parser.add_argument("--regularization", default=1e-3, type=float)
  parser.add_argument("--cuda", action='store_true', dest="cuda")
  parser.set_defaults(cuda=False)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps_demosaicking')

  main(args)

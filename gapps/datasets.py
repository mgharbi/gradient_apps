import logging
import os
import struct
import re
import random
import hashlib
import sys

import numpy as np
import skimage.io
import skimage.transform
from scipy.misc import imread, imresize
from torchvision import transforms

import torch as th
from torch.utils.data import Dataset

import gapps.utils as utils

log = logging.getLogger("gapps")

class DeconvDataset(Dataset):
  def __init__(self, filelist, is_validate = False):
    self.root = os.path.dirname(filelist)
    with open(filelist) as fid:
      self.files = [l.strip() for l in fid.readlines()]
    if is_validate:
      random_files = []
      random.seed(8888)
      for i in range(8):
        random_files.append(self.files[random.randint(0, len(self.files)-1)])
      self.files = random_files
    else:
      random.seed(9999)
      random.shuffle(self.files)

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    kernel_size = 11
    # 20 pixels of boundaries since we are going to clamp them
    crop_size = [256 + 40, 256 + 40]
    try:
      reference = skimage.io.imread(os.path.join(os.path.join(self.root, "imagenet_raw"), self.files[idx]))
      #reference = skimage.io.imread(os.path.join(self.root, self.files[idx]))
      reference = reference.astype(np.float32)/255.0
  
      seed = int(hashlib.md5(self.files[idx].encode('utf-8')).hexdigest(), 16) % (1 << 32)
      np.random.seed(seed)
  
      # Randomly choose a crop if reference is larger than this
      reference_size = reference.shape
      if len(reference.shape) == 2:
        reference = np.stack([reference, reference, reference], axis=-1)
      if reference.shape[0] > crop_size[0] and reference.shape[1] > crop_size[1]:
        left_top = [np.random.randint(0, reference_size[0] - crop_size[0]),
                    np.random.randint(0, reference_size[1] - crop_size[1])]
        reference = reference[left_top[0]:left_top[0]+crop_size[0],
                              left_top[1]:left_top[1]+crop_size[1],
                              :]
      else:
        # otherwise resize
        tmp = np.zeros([crop_size[0], crop_size[1], 3], dtype=np.float32)
        w = min(reference.shape[0], crop_size[0])
        h = min(reference.shape[1], crop_size[1])
        tmp[0:w, 0:h, :] = reference[0:w, 0:h, :]
        reference = tmp
    except:
      print('Couldn\'t load ', self.files[idx])
      reference = np.zeros([crop_size[0], crop_size[1], 3], dtype=np.float32)

    psf = utils.sample_psf(kernel_size)
    blurred = utils.make_blur(reference, psf, 0.01)
    reference = reference.transpose((2, 0, 1))
    blurred = blurred.transpose((2, 0, 1))
    return blurred, reference, psf

class DenoiseDataset(Dataset):
  def __init__(self, filelist, is_validate = False):
    self.root = os.path.dirname(filelist)
    with open(filelist) as fid:
      self.files = [l.strip() for l in fid.readlines()]
    if is_validate:
      random_files = []
      random.seed(8891)
      for i in range(8):
        random_files.append(self.files[random.randint(0, len(self.files)-1)])
      self.files = random_files
    else:
      random.seed(9999)
      random.shuffle(self.files)

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    # 5 pixels of boundaries since we are going to clamp them
    crop_size = [256 + 10, 256 + 10]
    try:
      reference = skimage.io.imread(os.path.join(os.path.join(self.root, "imagenet_raw"), self.files[idx]))
      #reference = skimage.io.imread(os.path.join(self.root, self.files[idx]))
      reference = reference.astype(np.float32)/255.0
  
      seed = int(hashlib.md5(self.files[idx].encode('utf-8')).hexdigest(), 16) % (1 << 32)
      np.random.seed(seed)
  
      # Randomly choose a crop if reference is larger than this
      reference_size = reference.shape
      if len(reference.shape) == 2:
        reference = np.stack([reference, reference, reference], axis=-1)
      if reference.shape[0] > crop_size[0] and reference.shape[1] > crop_size[1]:
        left_top = [np.random.randint(0, reference_size[0] - crop_size[0]),
                    np.random.randint(0, reference_size[1] - crop_size[1])]
        reference = reference[left_top[0]:left_top[0]+crop_size[0],
                              left_top[1]:left_top[1]+crop_size[1],
                              :]
      else:
        # otherwise resize
        tmp = np.zeros([crop_size[0], crop_size[1], 3], dtype=np.float32)
        w = min(reference.shape[0], crop_size[0])
        h = min(reference.shape[1], crop_size[1])
        tmp[0:w, 0:h, :] = reference[0:w, 0:h, :]
        reference = tmp
    except:
      exc_type, exc_value, exc_traceback = sys.exc_info()
      print("*** print_tb:")
      traceback.print_tb(exc_traceback, limit=5, file=sys.stdout)
      print('Couldn\'t load ', self.files[idx])
      reference = np.zeros([crop_size[0], crop_size[1], 3], dtype=np.float32)

    noisy = utils.make_noisy(reference, 0.1)
    reference = reference.transpose((2, 0, 1))
    noisy = noisy.transpose((2, 0, 1))
    return noisy, reference

class DemosaickingDataset(Dataset):
  def __init__(self, filelist, transform=None):
    self.transform=transform
    self.num_inputs = 1
    self.num_targets = 1

    self.root = os.path.dirname(filelist)
    with open(filelist) as fid:
      self.files = [l.strip() for l in fid.readlines()]

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    reference = skimage.io.imread(os.path.join(self.root, self.files[idx]))
    reference = reference.astype(np.float32)/255.0
    mosaick = utils.make_mosaick(reference)
    reference = reference.transpose((2, 0, 1))
    mosaick = np.expand_dims(mosaick, 0)

    return mosaick, reference
    # return mosaick, reference[1:2, ...]

class ADESegmentationDataset(Dataset):
  def __init__(self, txt, opt, max_sample=-1, is_train=1):
    self.root_img = opt.root_img
    self.root_seg = opt.root_seg
    self.imgSize = opt.imgSize
    self.segSize = opt.segSize
    self.is_train = is_train

    # mean and std
    self.img_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    self.list_sample = [x.rstrip() for x in open(txt, 'r')]

    if self.is_train:
      random.shuffle(self.list_sample)
    if max_sample > 0:
      self.list_sample = self.list_sample[0:max_sample]
    num_sample = len(self.list_sample)
    assert num_sample > 0
    print('# samples: {}'.format(num_sample))

  def _scale_and_crop(self, img, seg, cropSize, is_train):
    h, w = img.shape[0], img.shape[1]

    if is_train:
      # random scale
      scale = random.random() + 0.5     # 0.5-1.5
      scale = max(scale, 1. * cropSize / (min(h, w) - 1))
    else:
      # scale to crop size
      scale = 1. * cropSize / (min(h, w) - 1)

    img_scale = imresize(img, scale, interp='bilinear')
    seg_scale = imresize(seg, scale, interp='nearest')

    h_s, w_s = img_scale.shape[0], img_scale.shape[1]
    if is_train:
      # random crop
      x1 = random.randint(0, w_s - cropSize)
      y1 = random.randint(0, h_s - cropSize)
    else:
      # center crop
      x1 = (w_s - cropSize) // 2
      y1 = (h_s - cropSize) // 2

    img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
    seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
    return img_crop, seg_crop

  def _flip(self, img, seg):
    img_flip = img[:, ::-1, :]
    seg_flip = seg[:, ::-1]
    return img_flip, seg_flip

  def __getitem__(self, index):
    img_basename = self.list_sample[index]
    path_img = os.path.join(self.root_img, img_basename)
    path_seg = os.path.join(self.root_seg,
                            img_basename.replace('.jpg', '.png'))

    assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)
    assert os.path.exists(path_seg), '[{}] does not exist'.format(path_seg)

    # load image and label
    try:
      img = imread(path_img, mode='RGB')
      seg = imread(path_seg)
      assert(img.ndim == 3)
      assert(seg.ndim == 2)
      assert(img.shape[0] == seg.shape[0])
      assert(img.shape[1] == seg.shape[1])

      # random scale, crop, flip
      if self.imgSize > 0:
        img, seg = self._scale_and_crop(img, seg,
                                        self.imgSize, self.is_train)
        if random.choice([-1, 1]) > 0:
          img, seg = self._flip(img, seg)

      # image to float
      img = img.astype(np.float32) / 255.
      img = img.transpose((2, 0, 1))

      if self.segSize > 0:
        seg = imresize(seg, (self.segSize, self.segSize),
                       interp='nearest')

      # label to int from -1 to 149
      seg = seg.astype(np.int) - 1

      # to torch tensor
      image = th.from_numpy(img)
      segmentation = th.from_numpy(seg)
    except Exception as e:
      print('Failed loading image/segmentation [{}]: {}'
            .format(path_img, e))
      # dummy data
      image = th.zeros(3, self.imgSize, self.imgSize)
      segmentation = -1 * th.ones(self.segSize, self.segSize).long()
      return image, segmentation, img_basename

    # substracted by mean and divided by std
    image = self.img_transform(image)

    return image, segmentation, img_basename

  def __len__(self):
    return len(self.list_sample)


class ToBatch(object):
  def __call__(self, sample):
    for k in sample.keys():
      if type(sample[k]) == np.ndarray:
        sample[k] = np.expand_dims(sample[k], 0)
    return sample


class ToTensor(object):
  def __call__(self, sample):
    for k in sample.keys():
      if type(sample[k]) == np.ndarray:
        sample[k] = th.from_numpy(sample[k])
    return sample

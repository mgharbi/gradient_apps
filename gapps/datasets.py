import logging
import os
import struct
import re
import random
import hashlib

import numpy as np
import skimage.io
import skimage.transform

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
        reference = np.resize(reference, [crop_size[0], crop_size[1], 3])
    except:
      print('Couldn\'t load ', self.files[idx])
      reference = np.zeros([crop_size[0], crop_size[1], 3], dtype=np.float32)

    psf = utils.sample_psf(kernel_size)
    blurred = utils.make_blur(reference, psf, 0.01)
    reference = reference.transpose((2, 0, 1))
    blurred = blurred.transpose((2, 0, 1))
    return blurred, reference, psf

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

    return mosaick, reference[1:2, ...]

class ADESegmentationDataset(Dataset):
  def __init__(self, root, transform=None):
    self.transform = transform
    self.root = root
    self._load_info(root)

    input_files = [f for f in os.listdir(os.path.join(root, "images")) if ".jpg" in f]
    input_files = sorted(input_files)

    output_files = [f.replace(".jpg", ".png") for f in input_files]
    input_files = [os.path.join(root, "images", f) for f in input_files]
    output_files = [os.path.join(root, "annotations", f) for f in output_files]

    for i, o in zip(input_files, output_files):
      if not os.path.exists(i):
        log.error("input file {} does not exists".format(i))
      if not os.path.exists(o):
        log.error("output file {} does not exists".format(o))

    self.count = len(input_files)
    self.input_files = input_files
    self.output_files = output_files

  def _load_info(self, root):
    self.ratios = []
    self.names = []
    with open(os.path.join(root, "objectInfo150.txt"), 'r') as fid:
      for i, l in enumerate(fid.xreadlines()):
        if i == 0:
          log.info("Loading datset info with fields {}".format(l.split()))
        else:
          data = l.split()
          # idx, ratio, train, val, name
          idx = int(data[0])
          ratio = float(data[1])
          train = int(data[2])
          val = int(data[3])
          name = str(" ".join(data[4:]))

          self.ratios.append(ratio)
          self.names.append(name)

  def __len__(self):
    return self.count

  def __getitem__(self, idx):
    input_im = skimage.io.imread(self.input_files[idx])
    label = skimage.io.imread(self.output_files[idx])

    input_im = input_im.astype(np.float)/255.0

    size = 256
    sz = input_im.shape
    ratio = max(size*1.0/sz[0], size*1.0/sz[1])
    new_height = int(np.ceil(ratio*sz[0]))
    new_width = int(np.ceil(ratio*sz[1]))

    # TODO: data aug
    input_im = input_im[:size, :size, :]
    label = label[:size, :size]

    # input_im = np.transpose(input_im, [2, 0, 1])

    sample = {"input": input_im, "label": label}
    if self.transform is not None:
      sample = self.transform(sample)

    return sample


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

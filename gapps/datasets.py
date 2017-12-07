import logging
import os
import struct
import re

import numpy as np
import skimage.io
import skimage.transform

import torch as th
from torch.utils.data import Dataset

log = logging.getLogger("gapps")

class DemosaickingDataset(Dataset):
  def __init__(self, root, transform=None):
    self.transform=transform
    self.num_inputs = 1
    self.num_targets = 1

  def __len__(self):
    return 100

  def __getitem__(self, idx):
    mosaick = np.zeros((1, 64, 64), dtype=np.float32)
    reference = np.zeros((3, 64, 64), dtype=np.float32)

    # if self.transform is not None:
    #   mosaick = self.transform(mosaick)
    #   reference = self.transform(reference)

    return mosaick, reference

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

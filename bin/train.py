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

import torchsample.modules as ts
from torchvision import transforms

import gapps.datasets as datasets
import gapps.modules as models

log = logging.getLogger("gapps")

def main(args):
  print "ok"
  model = models.LearnableDemosaick()
  trainer = ts.ModuleTrainer(model)
  trainer.compile(loss=th.nn.MSELoss(), optimizer="adam")

  dset = datasets.DemosaickingDataset("tmp", transform=datasets.ToTensor())
  loader = DataLoader(dset, batch_size=32)

  trainer.fit_loader(loader, num_epoch=1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)
  setproctitle.setproctitle('gapps')

  main(args)

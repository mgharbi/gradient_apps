import torch as th
from torch.autograd import Variable

import torchlib.viz as viz

def forward_with_loss(batch, model, crit, cuda=False):
  (imgs, segs, infos) = batch

  # feed input data
  input_img = Variable(imgs)
  label_seg = Variable(segs)
  if cuda:
    input_img = input_img.cuda()
    label_seg = label_seg.cuda()

  pred = model(input_img)
  loss = crit(pred, label_seg)

  return pred, loss

class Callback(object):
  def __init__(self, model, num_batches, val_loader, env=None):
    self.model = model
    self.num_batches = num_batches
    self.val_loader = val_loader

    self.loss_viz = viz.ScalarVisualizer(
        "loss", opts={"legend": ["train", "val"]}, env=env)
    self.accuracy_viz = viz.ScalarVisualizer(
        "accuracy", opts={"legend": ["train", "val"]}, env=env)

    self.current_epoch = 0

  def on_epoch_begin(self, epoch):
    self.current_epoch = epoch

  def on_batch_end(self, batch, logs):
    frac = self.current_epoch + batch*1.0/self.num_batches
    if "loss" in logs.keys():
      self.loss_viz.update(frac, logs['loss'], name="train")
    if "acc" in logs.keys():
      self.accuracy_viz.update(frac, logs['acc'], name="train")

  def on_epoch_end(self, epoch, logs):
    if "loss" in logs.keys():
      self.loss_viz.update(epoch+1, logs['loss'], name="val")
    if "acc" in logs.keys():
      self.accuracy_viz.update(epoch+1, logs['acc'], name="val")

import torch as th
from torch.autograd import Variable

def linear_slice():
  """
  We want to fech data points from "src_data" and put them in "output" so 
  that output[i] ~ src_data[lookup_coord[i]], where lookup_coord is an array
  specifying the location of the requested points.
      
  The values in "src_data" are linearly interpolated, so the mapping is
  differentiable a.e.
  """

  dst_sz = 8
  src_sz = 4

  # Some random data we want to interpolate from
  src_data = Variable(th.rand(src_sz), requires_grad=True)

  # interpolation coordinate w.r.t. src_data, in [0, src_sz[
  lookup_coord = Variable(th.rand(dst_sz) * src_sz, requires_grad=True)

  #      lower     coord  upper
  # -------+--------*------+---
  lower_bin = th.clamp(th.floor(lookup_coord-0.5), min=0)
  upper_bin = th.clamp(lower_bin+1, max=src_sz-1)  # make sure we're in bounds

  # Linear interpolation weight
  weight = th.abs(lookup_coord-0.5 - lower_bin)

  # Make the coordinates integers to allow indexing
  lower_bin = lower_bin.long()
  upper_bin = upper_bin.long()

  # Interpolate the data from src_data
  output = src_data[lower_bin]*(1.0 - weight) + src_data[upper_bin]*weight

  # Backprop
  loss = output.sum()
  loss.backward()

  # Check the gradients
  print(src_data.grad)
  print(lookup_coord.grad)

  # We also want to write at locations indexed by an array
  data_copy = src_data + 1
  data_copy[lower_bin] += 1.0
  loss = data_copy.sum()
  loss.backward()


if __name__ == "__main__":
  linear_slice()

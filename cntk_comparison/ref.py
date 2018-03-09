import numpy as np
import torch as th
from torch.autograd import Variable
import time

def fwd(guide, grid, sigma_s, sigma_r):
  bs, sz, _ = guide.shape
  _, n_chans, small_sz, _ ,_ = grid.shape
  xx, yy = np.meshgrid(np.arange(0, sz), np.arange(0, sz))

  # start = time.time()
  guide = guide.unsqueeze(1)

  # Slice
  gx = ((xx+0.5)/sz) * small_sz
  gy = ((yy+0.5)/sz) * small_sz
  gz = guide*sigma_r

  # Enclosing cell
  fx = np.maximum(np.floor(gx - 0.5).astype(np.int64), 0)
  fy = np.maximum(np.floor(gy - 0.5).astype(np.int64), 0)
  fz = th.clamp(th.floor(gz-0.5), min=0)
  cx = np.minimum(fx+1, small_sz-1)
  cy = np.minimum(fy+1, small_sz-1)
  cz = th.clamp(fz+1, max=sigma_r-1)

  # Trilerp weights
  wx = Variable(th.from_numpy((gx - 0.5 - fx).astype(np.float32)).cuda())
  wy = Variable(th.from_numpy((gy - 0.5 - fy).astype(np.float32)).cuda())
  wz = th.abs(gz-0.5 - fz)

  # Make indices broadcastable
  # fx = np.expand_dims(fx, 0)
  # fy = np.expand_dims(fy, 0)
  fz = fz.long()[:, 0].view(bs, 1, sz, sz)
  cz = cz.long()[:, 0].view(bs, 1, sz, sz)

  batch_idx = th.from_numpy(np.arange(bs)).view(bs, 1, 1, 1).cuda()
  c_idx = th.from_numpy(np.arange(n_chans)).view(1, n_chans, 1, 1).cuda()

  out = grid[batch_idx, c_idx, fy, fx, fz]*(1-wx)*(1-wy)*(1-wz) + \
        grid[batch_idx, c_idx, fy, fx, cz]*(1-wx)*(1-wy)*(  wz) + \
        grid[batch_idx, c_idx, cy, fx, fz]*(1-wx)*(  wy)*(1-wz) + \
        grid[batch_idx, c_idx, cy, fx, cz]*(1-wx)*(  wy)*(  wz) + \
        grid[batch_idx, c_idx, fy, cx, fz]*(  wx)*(1-wy)*(1-wz) + \
        grid[batch_idx, c_idx, fy, cx, cz]*(  wx)*(1-wy)*(  wz) + \
        grid[batch_idx, c_idx, cy, cx, fz]*(  wx)*(  wy)*(1-wz) + \
        grid[batch_idx, c_idx, cy, cx, cz]*(  wx)*(  wy)*(  wz)
  # elapsed = (time.time() - start)*1000
  # print("runtime {}ms".format(elapsed))

  return out

def fwd_ref(input, guide, grid, sigma_s, sigma_r):
      # Get input dimensions
      bs, ci, h, w = input.shape
      _, c, gd, gh, gw = grid.shape

      # Coordinates in the fullres image
      xx = Variable(th.arange(0, w).view(1, -1).repeat(h, 1))
      yy = Variable(th.arange(0, h).view(-1, 1).repeat(1, w))

      if input.is_cuda:
        xx = xx.cuda()
        yy = yy.cuda()
      # Spatial coordinates in the bilateral grid 
      gx = ((xx+0.5)/w) * gw
      gy = ((yy+0.5)/h) * gh
      gz = th.clamp(guide, 0.0, 1.0)*gd

      # Coordinates of the neighboring grid voxels
      fx = th.clamp(th.floor(gx - 0.5), min=0)
      fy = th.clamp(th.floor(gy - 0.5), min=0)
      fz = th.clamp(th.floor(gz-0.5), min=0)

      # Interpolation weights
      wx = gx - 0.5 - fx
      wy = gy - 0.5 - fy
      wx = wx.unsqueeze(0).unsqueeze(0)
      wy = wy.unsqueeze(0).unsqueeze(0)
      wz = th.abs(gz-0.5 - fz)
      wz = wz.unsqueeze(1)

      # Make the voxel coordinates integers to be use in slicing
      fx = fx.long().unsqueeze(0).unsqueeze(0)
      fy = fy.long().unsqueeze(0).unsqueeze(0)
      fz = fz.long()
      cx = th.clamp(fx+1, max=gw-1);
      cy = th.clamp(fy+1, max=gh-1);
      cz = th.clamp(fz+1, max=gd-1)

      # Make indices broadcastable
      fz = fz.view(bs, 1, h, w)
      cz = cz.view(bs, 1, h, w)

      # Indices to slice along the batch axis
      batch_idx = th.arange(bs).view(bs, 1, 1, 1).long()
      if gz.is_cuda:
        batch_idx = batch_idx.cuda()
      out = []
      # Number of output channels
      co = c // (ci+1)
      # Construct the output channels, one at a time
      for c_ in range(co):
        # Select the relevant affine coefficients in the grid
        c_idx = th.arange((ci+1)*c_, (ci+1)*(c_+1)).view(1, ci+1, 1, 1).long()
        if gz.is_cuda:
          c_idx = c_idx.cuda()
        # Slice to upsample them to full-res
        a = grid[batch_idx, c_idx, fz, fy, fx]*(1-wx)*(1-wy)*(1-wz) + \
                 grid[batch_idx, c_idx, cz, fy, fx]*(1-wx)*(1-wy)*(  wz) + \
                 grid[batch_idx, c_idx, fz, cy, fx]*(1-wx)*(  wy)*(1-wz) + \
                 grid[batch_idx, c_idx, cz, cy, fx]*(1-wx)*(  wy)*(  wz) + \
                 grid[batch_idx, c_idx, fz, fy, cx]*(  wx)*(1-wy)*(1-wz) + \
                 grid[batch_idx, c_idx, cz, fy, cx]*(  wx)*(1-wy)*(  wz) + \
                 grid[batch_idx, c_idx, fz, cy, cx]*(  wx)*(  wy)*(1-wz) + \
                 grid[batch_idx, c_idx, cz, cy, cx]*(  wx)*(  wy)*(  wz)

        # Construct the output channel as an affine combination of input channels
        o = th.sum(a[:, :-1, ...]*input, 1) + a[:, -1, ...]
        out.append(o.unsqueeze(1))
      out = th.cat(out, 1)
      # Assemble all the output channels in a single tensor
      return out

def main():
  bs = 4
  n_chans = 3

  sigma_s = 16
  sigma_r = 8

  # 4x4x1024x1024
  # 4x12x64x64

  sz = 1024
  # sz = 1024
  small_sz = sz // sigma_s

  input = th.rand(bs, n_chans, sz, sz)
  guide = th.rand(bs, sz, sz)
  grid = th.rand(bs, n_chans*(n_chans+1), sigma_r, small_sz, small_sz)

  input = Variable(input, requires_grad=True)
  guide = Variable(guide, requires_grad=True)
  grid = Variable(grid, requires_grad=True)

  input = input.cuda()
  guide = guide.cuda()
  grid = grid.cuda()

  n = 8
  for i in range(n):
    if i == 1:
      start = time.time()

    out = fwd_ref(input, guide, grid, sigma_s, sigma_r)
    loss = out.sum()
    loss.backward()

  elapsed = (time.time() - start)*1000
  elapsed /= n-1
  print("runtime {}ms".format(elapsed))

  print(out.shape)

if __name__ == "__main__":
  main()

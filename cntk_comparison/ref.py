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

def main():
  bs = 4
  n_chans = 4

  sigma_s = 16
  sigma_r = 12

  # 4x4x1024x1024
  # 4x12x64x64

  sz = 1024
  # sz = 1024
  small_sz = sz // sigma_s

  guide = th.rand(bs, sz, sz)
  grid = th.rand(bs, n_chans, small_sz, small_sz, sigma_r)

  guide = Variable(guide, requires_grad=True)
  grid = Variable(grid, requires_grad=True)

  guide = guide.cuda()
  grid = grid.cuda()

  n = 8
  for i in range(n):
    if i == 1:
      start = time.time()

    out = fwd(guide, grid, sigma_s, sigma_r)
    loss = out.sum()
    loss.backward()

  elapsed = (time.time() - start)*1000
  elapsed /= n-1
  print("runtime {}ms".format(elapsed))

  print(out.shape)

if __name__ == "__main__":
  main()

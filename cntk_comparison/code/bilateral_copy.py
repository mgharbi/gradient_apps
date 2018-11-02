import cntk as C
import numpy as np
import time
import skimage.io as skio

def main():
  show_image = False
  if show_image:
    bs = 1
    ci = 3
    co = 3
    cg = co*(ci+1)
    gd = 8
    gh = 64
    gw = 64
    h = 256
    w = 256
  else:
    bs = 1
    ci = 3
    co = 3
    cg = co*(ci+1)
    gd = 8
    gh = 64
    gw = 64
    h = 1024
    w = 1024

  im = C.input_variable([bs, ci, h, w], needs_gradient=True, dynamic_axes=[])
  guide = C.input_variable([bs, h, w], needs_gradient=True, dynamic_axes=[])
  guide_no_grad = C.input_variable([bs, h, w], needs_gradient=False, dynamic_axes=[])
  grid = C.input_variable([bs, cg, gd, gh, gw], needs_gradient=True, dynamic_axes=[])
  # Create indices
  xx = np.arange(0, w).reshape(1, -1).repeat(h, 0).astype(np.float32)
  yy = np.arange(0, h).reshape(-1, 1).repeat(w, 1).astype(np.float32)
  xx = C.Constant(xx, xx.shape)
  yy = C.Constant(yy, yy.shape)
  gx = ((xx+0.5)/w) * gw
  gy = ((yy+0.5)/h) * gh
  gz = C.clip(guide, 0.0, 1.0) * gd
  gz_no_grad = C.clip(guide_no_grad, 0.0, 1.0) * gd
  fx = C.element_max(C.floor(gx-0.5), 0.0)
  fy = C.element_max(C.floor(gy-0.5), 0.0)
  fz = C.element_max(C.floor(gz-0.5), 0.0)
  fz_no_grad = C.element_max(C.floor(gz_no_grad-0.5), 0.0)
  wx = gx-0.5-fx
  wy = gy-0.5-fy
  wx = C.expand_dims(C.expand_dims(wx, -1-len(wx.shape)), -1-len(wx.shape))
  wy = C.expand_dims(C.expand_dims(wy, -1-len(wy.shape)), -1-len(wy.shape))
  wz = C.abs(gz-0.5-fz)
  wz = C.expand_dims(wz, 0)
  fx = C.expand_dims(C.expand_dims(fx, -1-len(fx.shape)), -1-len(fx.shape))
  fy = C.expand_dims(C.expand_dims(fy, -1-len(fy.shape)), -1-len(fy.shape))
  cx = C.element_min(fx+1, gw-1)
  cy = C.element_min(fy+1, gh-1)
  cz = C.element_min(fz_no_grad+1, gd-1)
  batch_idx = np.arange(bs).reshape(bs, 1, 1, 1).astype(np.float32)
  batch_idx = C.Constant(batch_idx, batch_idx.shape)
  out = []
  flat_grid = C.reshape(grid, [-1])
  for c_ in range(co):
    c_idx = np.arange((ci+1)*c_, (ci+1)*(c_+1)).reshape(1, ci+1, 1, 1).astype(np.float32)
    c_idx = C.Constant(c_idx, c_idx.shape)
    def flatten_and_gather(x, y, z):
      linear_idx = x+gw*y+gw*gh*z+c_idx*gw*gh*gd+batch_idx*gw*gh*gd*cg
      flat_linear_idx = C.reshape(linear_idx, [-1])
      return C.reshape(C.gather(flat_grid, flat_linear_idx), linear_idx.shape)
    gather_fff = flatten_and_gather(fx, fy, fz_no_grad)
    gather_ffc = flatten_and_gather(fx, fy, cz)
    gather_fcf = flatten_and_gather(fx, cy, fz_no_grad)
    gather_fcc = flatten_and_gather(fx, cy, cz)
    gather_cff = flatten_and_gather(cx, fy, fz_no_grad)
    gather_cfc = flatten_and_gather(cx, fy, cz)
    gather_ccf = flatten_and_gather(cx, cy, fz_no_grad)
    gather_ccc = flatten_and_gather(cx, cy, cz)
    a = gather_fff*(1-wx)*(1-wy)*(1-wz) + \
        gather_ffc*(1-wx)*(1-wy)*(  wz) + \
        gather_fcf*(1-wx)*(  wy)*(1-wz) + \
        gather_fcc*(1-wx)*(  wy)*(  wz) + \
        gather_cff*(  wx)*(1-wy)*(1-wz) + \
        gather_cfc*(  wx)*(1-wy)*(  wz) + \
        gather_ccf*(  wx)*(  wy)*(1-wz) + \
        gather_ccc*(  wx)*(  wy)*(  wz)
    o = C.reduce_sum(a[:, :-1, ...] * im, 1) + a[:, -1, ...]
    print(o.shape)
    out.append(C.expand_dims(o, 0))
  out = C.splice(*out, axis=1)
  loss = C.reduce_l2(out)

  grid_val = np.random.rand(bs, cg, gd, gh, gw).astype(np.float32)
  if show_image:
    guide_val = skio.imread("/data/rgb.png").mean(2)[:h, :w].astype(np.float32)
    guide_val = np.expand_dims(guide_val / 255.0, 0)
    im_val = np.tile(np.expand_dims(guide_val, 1), [1, 3, 1, 1])
    out_val = out.eval({im : im_val, guide : guide_val, guide_no_grad : guide_val, grid : grid_val})
    out_val = np.clip(np.transpose(np.squeeze(out_val), [1, 2, 0]), 0, 1)
    skio.imsave("/output/imout.png", out_val)
  else:
    im_val = np.random.randn(bs, ci, h, w)
    guide_val = np.random.rand(bs, h, w).astype(np.float32)
    # burning iteration
    for it in range(5):
      print('burning (', it, ')')
      g = loss.grad({im : im_val, guide : guide_val, guide_no_grad : guide_val, grid : grid_val})
    # actual iterations
    start = time.time()
    for it in range(50):
      print('profiling (', it, ')')
      g = loss.grad({im : im_val, guide : guide_val, guide_no_grad : guide_val, grid : grid_val})
    end = time.time()
  runtime = (end-start)*1000.0/50.0
  print('Runtime:', runtime)

  #print(g)

if __name__ == "__main__":
  main()

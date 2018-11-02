import cntk as C
import numpy as np
import time
import skimage.io as skio

def main():
  bs = 4
  c = 64
  h = 512
  w = 512

  im = C.input_variable([bs, c, h, w], needs_gradient=True, dynamic_axes=[])
  warp = C.input_variable([bs, 2, h, w], needs_gradient=True, dynamic_axes=[])
  warp_ng = C.input_variable([bs, 2, h, w], needs_gradient=False, dynamic_axes=[])
  # Create indices
  dx = 0.5 * (warp[:, 0, :, :] + 1.0)
  dy = 0.5 * (warp[:, 1, :, :] + 1.0)
  new_x = C.clip(dx * w, 0, w)
  new_y = C.clip(dy * h, 0, h)
  fx = C.clip(C.floor(new_x), 0, w - 2)
  fy = C.clip(C.floor(new_y), 0, h - 2)
  wx = new_x - fx
  wy = new_y - fy
  dx_ng = 0.5 * (warp_ng[:, 0, :, :] + 1.0)
  dy_ng = 0.5 * (warp_ng[:, 1, :, :] + 1.0)
  new_x_ng = C.clip(dx_ng * w, 0, w)
  new_y_ng = C.clip(dy_ng * h, 0, h)
  fx_ng = C.clip(C.floor(new_x_ng), 0, w - 2)
  fy_ng = C.clip(C.floor(new_y_ng), 0, h - 2)

  chan_idx = np.arange(c).reshape(1, c, 1, 1)
  chan_idx = C.Constant(chan_idx, chan_idx.shape)
  batch_idx = np.arange(bs).reshape(bs, 1, 1, 1)
  batch_idx = C.Constant(batch_idx, batch_idx.shape)
  flat_im = C.reshape(im, [-1])
  def flatten_and_gather(x, y):
    linear_idx = x + w*y + w*h*chan_idx + w*h*c*batch_idx
    flat_linear_idx = C.reshape(linear_idx, [-1])
    return C.reshape(C.gather(flat_im, flat_linear_idx),linear_idx.shape)
  gather_ff = flatten_and_gather(fx_ng    , fy_ng    )
  gather_fc = flatten_and_gather(fx_ng    , fy_ng + 1)
  gather_cf = flatten_and_gather(fx_ng + 1, fy_ng    )
  gather_cc = flatten_and_gather(fx_ng + 1, fy_ng + 1)
  out = gather_ff*(1-wx)*(1-wy) + \
        gather_fc*(1-wx)*(  wy) + \
        gather_cf*(  wx)*(1-wy) + \
        gather_cc*(  wx)*(  wy)
  loss = C.reduce_l2(out)

  im_val = np.random.randn(bs, c, h, w).astype(np.float32)
  warp_val = np.random.rand(bs, 2, h, w).astype(np.float32)
  # burning iteration
  for it in range(5):
    print('burning (', it, ')')
    g = loss.grad({im : im_val, warp : warp_val, warp_ng : warp_val})
  # actual iterations
  start = time.time()
  for it in range(50):
    print('profiling (', it, ')')
    g = loss.grad({im : im_val, warp : warp_val, warp_ng : warp_val})
  end = time.time()
  runtime = (end-start)*1000.0/50.0
  print('Runtime:', runtime)

  #print(g)

if __name__ == "__main__":
  main()

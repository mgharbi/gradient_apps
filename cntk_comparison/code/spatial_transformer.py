import cntk as C
import numpy as np
import time
import skimage.io as skio

def main():
  bs = 4
  c = 16
  h = 512
  w = 512

  im = C.input_variable([bs, c, h, w], needs_gradient=True, dynamic_axes=[])
  affine_mtx = C.input_variable([bs, 2, 3], needs_gradient=True, dynamic_axes=[])
  affine_mtx_ng = C.input_variable([bs, 2, 3], needs_gradient=False, dynamic_axes=[])
  xx = np.arange(0, w).reshape(1, -1).repeat(h, 0).astype(np.float32)
  yy = np.arange(0, h).reshape(-1, 1).repeat(w, 1).astype(np.float32)
  xx = C.Constant(xx, xx.shape)
  yy = C.Constant(yy, yy.shape) 
  nrm_x = 2.0 * (xx / w) - 1.0
  nrm_y = 2.0 * (yy / h) - 1.0
  nrm_x = C.expand_dims(nrm_x, -1 - len(nrm_x.shape))
  nrm_y = C.expand_dims(nrm_y, -1 - len(nrm_y.shape))
  xformed_x = affine_mtx[:, 0, 0] * nrm_x + \
              affine_mtx[:, 0, 1] * nrm_y + \
              affine_mtx[:, 0, 2]
  xformed_y = affine_mtx[:, 1, 0] * nrm_x + \
              affine_mtx[:, 1, 1] * nrm_y + \
              affine_mtx[:, 1, 2]
  xformed_x = 0.5 * xformed_x + 1.0
  xformed_y = 0.5 * xformed_y + 1.0
  xformed_x = C.expand_dims(xformed_x, 0)
  xformed_y = C.expand_dims(xformed_y, 0)
  xformed_x_ng = affine_mtx_ng[:, 0, 0] * nrm_x + \
                 affine_mtx_ng[:, 0, 1] * nrm_y + \
                 affine_mtx_ng[:, 0, 2]
  xformed_y_ng = affine_mtx_ng[:, 1, 0] * nrm_x + \
                 affine_mtx_ng[:, 1, 1] * nrm_y + \
                 affine_mtx_ng[:, 1, 2]
  xformed_x_ng = C.expand_dims(xformed_x_ng, 0)
  xformed_y_ng = C.expand_dims(xformed_y_ng, 0)

  fx = C.clip(w * xformed_x, 0, w-2)
  fy = C.clip(h * xformed_y, 0, h-2)
  wx = xformed_x - fx
  wy = xformed_y - fy
  fx_ng = C.clip(w * xformed_x_ng, 0, w-2)
  fy_ng = C.clip(h * xformed_y_ng, 0, h-2)

  chan_idx = np.arange(c).reshape(1, c, 1, 1)
  chan_idx = C.Constant(chan_idx, chan_idx.shape)
  batch_idx = np.arange(bs).reshape(bs, 1, 1, 1)
  batch_idx = C.Constant(batch_idx, batch_idx.shape)
  flat_im = C.reshape(im, [-1])
  def flatten_and_gather(x, y):
    linear_idx = x + w*y
    linear_idx = linear_idx + w*h*chan_idx + w*h*c*batch_idx
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
  affine_mtx_val = np.zeros([bs, 2, 3], dtype=np.float32)
  affine_mtx_val[:, 0, 1] = 1.0
  affine_mtx_val[:, 1, 0] = 1.0
  # burning iteration
  for it in range(5):
    print('burning (', it, ')')
    g = loss.grad({im : im_val, affine_mtx : affine_mtx_val, affine_mtx_ng : affine_mtx_val})
  # actual iterations
  start = time.time()
  for it in range(50):
    print('profiling (', it, ')')
    g = loss.grad({im : im_val, affine_mtx : affine_mtx_val, affine_mtx_ng : affine_mtx_val})
  end = time.time()
  runtime = (end-start)*1000.0/50.0
  print('Runtime:', runtime)


if __name__ == "__main__":
  main()

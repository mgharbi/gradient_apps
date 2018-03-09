import cntk as C
import numpy as np
import time
import skimage.io as skio

def grid_coord(guide, xx, yy, sz, grid_sz, sigma_r):
  gx = ((xx+0.5)/sz) * grid_sz
  gy = ((yy+0.5)/sz) * grid_sz
  expanded_guide = C.reshape(guide, [1, sz, sz])
  gz = expanded_guide*sigma_r
  fx = C.floor(gx - 0.5)
  fy = C.floor(gy - 0.5)
  fz = C.clip(C.floor(gz - 0.5), 0, sigma_r-1)
  cx = C.element_min(fx+1, grid_sz-1)
  cy = C.element_min(fy+1, grid_sz-1)
  cz = C.clip(fz+1, 0, sigma_r-1)
  return gx, gy, gz, fx, fy, fz, cx, cy, cz

def BilateralSlice(sz, n_chans, grid_sz=16, sigma_r=8, sigma_s=16):
  grid = C.Parameter([n_chans, sigma_r, grid_sz, grid_sz], name="grid")

  # Flatten data for gather op
  flat_grid = C.reshape(grid, [grid_sz*grid_sz*sigma_r*n_chans])

  yy, xx = np.meshgrid(np.arange(0, sz), np.arange(0, sz))
  xx = np.expand_dims(xx, 0)
  yy = np.expand_dims(yy, 0)
  cc = np.arange(0, n_chans)
  cc = np.expand_dims(cc, 1)
  cc = np.expand_dims(cc, 2)
  xx = C.Constant(xx, xx.shape)
  yy = C.Constant(yy, yy.shape)
  cc = C.Constant(cc, cc.shape)

  @C.functions.BlockFunction("BilateralSlice", "bilateral_slice")
  def bilateral_slice(guide, guide_no_grad):
    gx_d, gy_d, gz_d, fx_d, fy_d, fz_d, _, _, _ = grid_coord(
        guide, xx, yy, sz, grid_sz, sigma_r)
    wx = (gx_d - 0.5 - fx_d)
    wy = (gy_d - 0.5 - fy_d)
    wz = C.abs(gz_d-0.5 - fz_d)

    # Enclosing cell
    gx, gy, gz, fx, fy, fz, cx, cy, cz = grid_coord(
        guide_no_grad, xx, yy, sz, grid_sz, sigma_r)

    output_components = []
    for ix, x in enumerate([fx, cx]):
      wx_ = (1-wx) if ix == 0 else wx
      for iy, y in enumerate([fy, cy]):
        wy_ = (1-wy) if iy == 0 else wy
        for iz, z in enumerate([fz, cz]):
          wz_ = (1-wz) if iz == 0 else wz
          linear_idx = x + grid_sz*(y + grid_sz*(z + sigma_r*cc))

          flat_linear_idx = C.reshape(linear_idx, [n_chans*sz*sz])

          # Slice
          interp = C.gather(flat_grid, flat_linear_idx)
          interp_fsz = C.reshape(interp, [n_chans, sz, sz])
          output_components.append(interp_fsz*wx_*wy_*wz_)

    out = sum(output_components)

    return out
  
  return bilateral_slice

GRID_SZ = 16
SZ = 128

def main():
  sz = 128
  N = 1
  n_chans = 1

  guide = C.input_variable([sz, sz], needs_gradient=True)
  guide_no_grad = C.input_variable([sz, sz], needs_gradient=False)
  model = BilateralSlice(sz, n_chans)
  print(model)

  out = model(guide, guide_no_grad)
  print(out)

  loss = C.squared_error(model(guide, guide_no_grad), guide_no_grad)
  data = np.random.uniform(size=[N, sz, sz]).astype(np.float32)

  print(out.forward({guide:data, guide_no_grad:data}))
  return

  # C.debugging.profiler.start_profiler("/output/pyprof")
  # C.debugging.profiler.enable_profiler()
  # learner = C.sgd(model.parameters, C.learning_parameter_schedule(0.01))
  # progress_writer = C.logging.ProgressPrinter(0)
  # print(model.parameters)
  # summary = loss.train((data, data), parameter_learners=[learner],
  #                      callbacks=[progress_writer], max_epochs=10,
  #                      minibatch_size=1)
  # C.debugging.profiler.stop_profiler()
  # # ---------------------------------------------------------------------------

  # learner = C.learners.sgd([grid], lr=C.learning_parameter_schedule(1e-1))
  # loss.train([guide_data], parameter_learners=[learner])
  # inputs = {grid:grid_data, guide:guide_data, guide_non_diff:guide_data}

  # # out_ = out.eval(inputs)
  # # out_ = np.transpose(np.squeeze(out_), [1, 2, 0])
  # # # out_ = np.squeeze(guide_data)
  # # print(out_.shape, out_.min(), out_.max())
  # # skio.imsave("/output/imout.png", out_)
  #
  # start = time.time()
  # for i in range(1):
  #   ret = loss.grad(inputs)
  # elapsed = (time.time() - start)*1000
  # print(elapsed)
  #
  # # for i in range(n):
  # #   if i == 1:
  # #     start = time.time()
  # #
  # #   ret = loss.grad(inputs)
  # #   # ret = grid.grad(inputs)
  # #   # ret = wy.eval(inputs)
  # #   # ret = loss.grad(inputs)
  # #   # ret = loss.eval(inputs)
  # # elapsed = (time.time() - start)*1000
  # # elapsed /= n-1
  # # print(ret)
  # print("runtime {}ms".format(elapsed))

if __name__ == "__main__":
  main()

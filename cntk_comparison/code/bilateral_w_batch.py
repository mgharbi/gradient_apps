import cntk as C
import numpy as np
import time
import skimage.io as skio

def grid_coord(guide, xx, yy, sz, small_sz, sigma_r, bs):
  gx = ((xx+0.5)/sz) * small_sz
  gy = ((yy+0.5)/sz) * small_sz
  expanded_guide = C.reshape(guide, [bs, 1, sz, sz])
  gz = expanded_guide*sigma_r
  fx = C.floor(gx - 0.5)
  fy = C.floor(gy - 0.5)
  fz = C.clip(C.floor(gz - 0.5), 0, sigma_r-1)
  cx = C.element_min(fx+1, small_sz-1)
  cy = C.element_min(fy+1, small_sz-1)
  cz = C.clip(fz+1, 0, sigma_r-1)
  return gx, gy, gz, fx, fy, fz, cx, cy, cz

def main():
  print("version", C.__version__)
  bs = 1
  n_chans = 1

  sigma_s = 16
  sigma_r = 12

  # 4x4x1024x1024
  # 4x12x64x64

  sz = 256
  # sz = 1024
  small_sz = sz // sigma_s

  yy, xx = np.meshgrid(np.arange(0, sz), np.arange(0, sz))
  cc, bb = np.meshgrid(np.arange(0, n_chans), np.arange(0, bs))

  xx = np.expand_dims(xx, 0)
  xx = np.expand_dims(xx, 0)
  yy = np.expand_dims(yy, 0)
  yy = np.expand_dims(yy, 0)

  bb = np.expand_dims(bb, 2)
  bb = np.expand_dims(bb, 3)
  cc = np.expand_dims(cc, 2)
  cc = np.expand_dims(cc, 3)

  # Compute graph
  grid = C.Parameter(
      [bs, n_chans, sigma_r, small_sz, small_sz],)
  # grid = C.input_variable(
  #     [bs, n_chans, sigma_r, small_sz, small_sz],
  #     dynamic_axes=[], needs_gradient=True)
  guide = C.input_variable([bs, sz, sz], dynamic_axes=[], needs_gradient=True)
  guide_non_diff = C.input_variable([bs, sz, sz], dynamic_axes=[])

  # Coordinates
  xx = C.Constant(xx, xx.shape)
  yy = C.Constant(yy, yy.shape)
  cc = C.Constant(cc, cc.shape)
  bb = C.Constant(bb, bb.shape)

  gx_d, gy_d, gz_d, fx_d, fy_d, fz_d, _, _, _ = grid_coord(
      guide, xx, yy, sz, small_sz, sigma_r, bs)

  # Trilerp weights
  wx = (gx_d - 0.5 - fx_d)
  wy = (gy_d - 0.5 - fy_d)
  wz = C.abs(gz_d-0.5 - fz_d)

  # Enclosing cell
  gx, gy, gz, fx, fy, fz, cx, cy, cz = grid_coord(
      guide_non_diff, xx, yy, sz, small_sz, sigma_r, bs)

  output_components = []
  for ix, x in enumerate([fx, cx]):
    wx_ = (1-wx) if ix == 0 else wx
    for iy, y in enumerate([fy, cy]):
      wy_ = (1-wy) if iy == 0 else wy
      for iz, z in enumerate([fz, cz]):
        wz_ = (1-wz) if iz == 0 else wz
        linear_idx = x + small_sz*(y + small_sz*(z + sigma_r*(cc + n_chans*bb)))

        # Flatten data for gather op
        flat_grid = C.reshape(grid, [bs*small_sz*small_sz*sigma_r*n_chans])
        flat_linear_idx = C.reshape(linear_idx, [bs*n_chans*sz*sz])

        # Slice
        interp = C.gather(flat_grid, flat_linear_idx)
        interp_fsz = C.reshape(interp, [bs, n_chans, sz, sz])
        output_components.append(interp_fsz*wz_*wx_*wy_)

  out = sum(output_components)
  loss = C.squared_error(out, guide)

  # svg = C.logging.graph.plot(out, "/output/graph.svg")

  grid_data = np.random.uniform(
      size=(bs, n_chans, sigma_r, small_sz, small_sz)).astype(np.float32)

  # guide_data = np.random.uniform(
  #     size=(bs, sz, sz)).astype(np.float32)
  guide_data = skio.imread("/data/rgb.png").mean(2)[:sz, :sz].astype(np.float32)
  guide_data = np.expand_dims(guide_data, 0) / 255.0

  inputs = {guide:guide_data, guide_non_diff:guide_data}

  # # --------FC ----------------------------------------------------------------
  # x = C.input_variable(3)
  # y = C.input_variable(2)
  # model = C.layers.Dense(2)
  # loss = C.cross_entropy_with_softmax(model(x), y)
  # print(loss)
  #
  # N = 128
  # labels = np.random.randint(0, 2, size=[N, 2]).astype(np.float32)
  # data = np.random.uniform(size=[N, 3]).astype(np.float32)
  #
  # C.debugging.profiler.start_profiler("/output/pyprof")
  # C.debugging.profiler.enable_profiler()
  # learner = C.sgd(model.parameters, C.learning_parameter_schedule(0.1))
  # progress_writer = C.logging.ProgressPrinter(0)
  # summary = loss.train((data, labels), parameter_learners=[learner],
  #                      callbacks=[progress_writer], max_epochs=10,
  #                      minibatch_size=1)
  # C.debugging.profiler.stop_profiler()
  # # ---------------------------------------------------------------------------

  # --------FC ----------------------------------------------------------------
  # x = C.input_variable((3, 16, 16))
  # # y = C.input_variable((3, 16, 16))
  # y = 2*x
  # model = C.layers.Convolution((3, 3), 3, pad=True)
  # loss = C.squared_error(model(x), y) / (16*16*3)
  #
  # N = 128
  # # labels = np.random.randint(0, 2, size=[N, 3, 16, 16]).astype(np.float32)
  # data = np.random.uniform(size=[N, 3, 16, 16]).astype(np.float32)
  #
  # C.debugging.profiler.start_profiler("/output/pyprof")
  # C.debugging.profiler.enable_profiler()
  # learner = C.sgd(model.parameters, C.learning_parameter_schedule(0.01))
  # progress_writer = C.logging.ProgressPrinter(0)
  # summary = loss.train((data,), parameter_learners=[learner],
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

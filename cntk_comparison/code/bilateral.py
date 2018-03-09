import cntk as C
import numpy as np
import time
import skimage.io as skio

def grid_coord(guide, xx, yy, sz, grid_sz, sigma_r):
  gx = ((xx+0.5)/sz) * grid_sz
  gy = ((yy+0.5)/sz) * grid_sz
  expanded_guide = C.reshape(guide, [1, sz, sz])
  gz = expanded_guide*sigma_r
  fx = C.clip(C.floor(gx - 0.5), 0, grid_sz-1)
  fy = C.clip(C.floor(gy - 0.5), 0, grid_sz-1)
  fz = C.clip(C.floor(gz - 0.5), 0, sigma_r-1)
  cx = C.element_min(fx+1, grid_sz-1)
  cy = C.element_min(fy+1, grid_sz-1)
  cz = C.clip(fz+1, 0, sigma_r-1)
  return gx, gy, gz, fx, fy, fz, cx, cy, cz


def BilateralSlice(sz, n_chans, grid_sz=64, sigma_r=12):
  gsize = [n_chans, sigma_r, grid_sz, grid_sz]
  grid = C.Parameter(gsize, 
                     name="grid", init=np.random.uniform(size=gsize))
  guide_scale = C.Parameter((1, ), 
                     name="guide_scale", init=np.ones((1, )))

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
    # Make sure we do sth that requires the gradient w.r.t guide
    scaled_guide = guide_scale*guide  
    gx_d, gy_d, gz_d, fx_d, fy_d, fz_d, _, _, _ = grid_coord(
        scaled_guide, xx, yy, sz, grid_sz, sigma_r)
    wx = C.abs(gx_d - 0.5 - fx_d)
    wy = C.abs(gy_d - 0.5 - fy_d)
    wz = C.abs(gz_d - 0.5 - fz_d)

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
          interp_fsz = C.reshape(interp, [n_chans, sz, sz])*wx_*wy_*wz_
          output_components.append(interp_fsz)

    out = sum(output_components)

    return out
  
  return bilateral_slice

def main():
  # C.device.try_set_default_device(C.device.cpu())
  show_image = False
  if show_image:
    sz = 256
    n_chans = 3
    bs = 1
    data = skio.imread("/data/rgb.png").mean(2)[:sz, :sz].astype(np.float32)
    data = np.expand_dims(data / 255.0, 0)
    n_epochs = 1000
    lr = 0.001
  else:
    sz = 1024
    n_chans = 4
    bs = 4
    N = 4
    data = np.random.uniform(size=[N, sz, sz]).astype(np.float32)
    n_epochs = 50
    lr = 0.000000001


  guide = C.input_variable([sz, sz], needs_gradient=True)
  guide_no_grad = C.input_variable([sz, sz], needs_gradient=False)
  model = BilateralSlice(sz, n_chans)
  out = model(guide, guide_no_grad)

  loss = C.squared_error(model(guide, guide_no_grad), guide_no_grad)

  start = time.time()
  n = 10
  for _ in range(n):
    inputs = {guide:data[0], guide_no_grad:data[0]}
    out_ = loss.grad(inputs)
  elapsed = (time.time() - start)*1000/n
  print("forward", elapsed, "ms/it")
  print(out_.shape)


  # --- Train -----------------------------------------------------------------
  start = time.time()
  C.debugging.profiler.start_profiler("/output/pyprof")
  C.debugging.profiler.enable_profiler()
  learner = C.sgd(model.parameters, C.learning_parameter_schedule(lr))
  progress_writer = C.logging.ProgressPrinter(0)
  print(model.parameters)
  summary = loss.train((data, data), parameter_learners=[learner],
                       callbacks=[progress_writer], max_epochs=n_epochs,
                       minibatch_size=bs)
  C.debugging.profiler.stop_profiler()
  elapsed = (time.time() - start)*1000/n_epochs
  print("training", elapsed, "ms/it")
  # ---------------------------------------------------------------------------


  # --- Show output -----------------------------------------------------------
  if show_image:
    inputs = {guide:data[0], guide_no_grad:data[0]}
    out_ = out.eval(inputs)
    print(out_.shape)
    out_ = np.clip(np.transpose(np.squeeze(out_), [1, 2, 0]), 0, 1)
    print(out_.shape, out_.min(), out_.max())
    skio.imsave("/output/imout.png", out_)
  # ---------------------------------------------------------------------------


if __name__ == "__main__":
  main()

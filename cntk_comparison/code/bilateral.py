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


def BilateralSlice(sz, i_chans, o_chans, grid_sz=64, sigma_r=8):
  gsize = [(i_chans+1)*o_chans, sigma_r, grid_sz, grid_sz]
  grid = C.Parameter(gsize, 
                     name="grid", init=np.random.uniform(size=gsize))
  guide_scale = C.Parameter((1, ), 
                     name="guide_scale", init=np.ones((1, )))
  grid_scale = C.Parameter((1, ), 
                     name="grid_scale", init=np.ones((1, )))
  im_scale = C.Parameter((1, ), 
                     name="im_scale", init=np.ones((1, )))


  yy, xx = np.meshgrid(np.arange(0, sz), np.arange(0, sz))
  xx = np.expand_dims(xx, 0)
  yy = np.expand_dims(yy, 0)
  cc = np.arange(0, i_chans+1)
  cc = np.expand_dims(cc, 1)
  cc = np.expand_dims(cc, 2)
  xx = C.Constant(xx, xx.shape)
  yy = C.Constant(yy, yy.shape)
  cc = C.Constant(cc, cc.shape)


  @C.functions.BlockFunction("BilateralSlice", "bilateral_slice")
  def bilateral_slice(im, guide, guide_no_grad):
    # Flatten data for gather op
    flat_grid = grid_scale*C.reshape(grid, [grid_sz*grid_sz*sigma_r*o_chans*(i_chans+1)])
    # flat_grid_u = C.unpack_batch(flat_grid)

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

    out_chans = []
    for chan in range(o_chans):
      output_components = []
      for ix, x in enumerate([fx, cx]):
        wx_ = (1-wx) if ix == 0 else wx
        for iy, y in enumerate([fy, cy]):
          wy_ = (1-wy) if iy == 0 else wy
          for iz, z in enumerate([fz, cz]):
            wz_ = (1-wz) if iz == 0 else wz

            linear_idx = x + grid_sz*(y + grid_sz*(z + sigma_r*(cc + chan*(i_chans+1))))
            flat_linear_idx = C.reshape(linear_idx, [(i_chans+1)*sz*sz])
            # Slice
            interp = C.gather(flat_grid, flat_linear_idx)
            interp_fsz = C.reshape(interp, [i_chans+1, sz, sz])*wx_*wy_*wz_
            output_components.append(interp_fsz)

      out_coeffs = sum(output_components)
      out_chan = C.reduce_sum(out_coeffs[:i_chans]*(im_scale*im) + out_coeffs[-1], 0)
      out_chans.append(out_chan)
    out = C.splice(*out_chans, axis=0)

    return out
  
  return bilateral_slice

def main():
  show_image = False
  sigma_r = 8
  grid_sz = 64
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
    n_chans = 3
    bs = 4
    N = 4
    data = np.random.uniform(size=[N, sz, sz]).astype(np.float32)
    n_epochs = 50
    lr = 0.000000001
  imdata = np.tile(np.expand_dims(data, 1), [1, n_chans, 1, 1])

  im = C.input_variable([n_chans, sz, sz], needs_gradient=True)
  guide = C.input_variable([sz, sz], needs_gradient=True)
  guide_no_grad = C.input_variable([sz, sz], needs_gradient=False)
  model = BilateralSlice(sz, n_chans, n_chans, sigma_r=sigma_r, grid_sz=grid_sz)
  out = model(im, guide, guide_no_grad)

  loss = C.squared_error(out, im)

  # --- Train -----------------------------------------------------------------
  C.debugging.profiler.start_profiler("/output/pyprof")
  C.debugging.profiler.enable_profiler()
  learner = C.sgd(model.parameters, C.learning_parameter_schedule(lr))
  progress_writer = C.logging.ProgressPrinter(0)
  summary = loss.train((imdata, data, data), parameter_learners=[learner],
                       callbacks=[progress_writer], max_epochs=n_epochs,
                       minibatch_size=bs)
  C.debugging.profiler.stop_profiler()
  # ---------------------------------------------------------------------------

  svg = C.logging.graph.plot(out, "/output/graph.svg")

  # --- Show output -----------------------------------------------------------
  if show_image:
    inputs = {guide:data[0], guide_no_grad:data[0]}
    out_ = out.eval(inputs)
    out_ = np.clip(np.transpose(np.squeeze(out_), [1, 2, 0]), 0, 1)
    skio.imsave("/output/imout.png", out_)
  # ---------------------------------------------------------------------------


if __name__ == "__main__":
  main()

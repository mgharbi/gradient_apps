#include "algorithms/spatial_transformer.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class SpatialTransformerForwardGenerator :
  public Generator<SpatialTransformerForwardGenerator> {
public:
  Input<Buffer<float>>  input{"input", 4};
  Input<Buffer<float>>  affine_mtx{"affine_mtx", 3};
  Output<Buffer<float>> output{"output", 4};

  void generate() {
    Func f_output = spatial_transformer(
        input, affine_mtx);
    output(x, y, c, n) = f_output(x, y, c, n);

    if (get_target().has_gpu_feature()) {
      Var xi, yi;
      // No need to use the autoscheduler on something so simple.
      output.compute_root().reorder(c, x, y, n).tile(x, y, xi, yi, 32, 8).gpu_blocks(x, y, n).gpu_threads(xi, yi);
    } else {
      output.compute_root().reorder(x, c, n, y).parallel(y).vectorize(x, 8);
    }
  }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::SpatialTransformerForwardGenerator,
    spatial_transformer_forward)

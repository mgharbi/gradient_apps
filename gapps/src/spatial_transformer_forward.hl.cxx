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

    SimpleAutoscheduleOptions options;
    options.gpu = get_target().has_gpu_feature();
    Func output_func = output;

#if 0
    std::set<std::string> dont_inline = {};
    simple_autoschedule(output_func,
        {
        {"input.min.0", 0},
        {"input.min.1", 0},
        {"input.min.2", 0},
        {"input.min.3", 0},
        {"input.extent.0", 512},
        {"input.extent.1", 512},
        {"input.extent.2", 16},
        {"input.extent.3", 8},
        {"affine_mtx.min.0", 0},
        {"affine_mtx.min.1", 0},
        {"affine_mtx.min.2", 0},
        {"affine_mtx.extent.0", 3},
        {"affine_mtx.extent.1", 2},
        {"affine_mtx.extent.2", 8},
        },
        {{0, 511},
          {0, 511},
          {0, 15},
          {0, 7}},
        options,
        dont_inline);
#endif
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

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

    std::set<std::string> dont_inline = {};

    simple_autoschedule(output_func,
        {
        {"input.min.0", 0},
        {"input.min.1", 0},
        {"input.min.2", 0},
        {"input.min.3", 0},
        {"input.extent.0", 256},
        {"input.extent.1", 256},
        {"input.extent.2", 3},
        {"input.extent.3", 16},
        {"affine_mtx.min.0", 0},
        {"affine_mtx.min.1", 0},
        {"affine_mtx.min.2", 0},
        {"affine_mtx.extent.0", 3},
        {"affine_mtx.extent.1", 2},
        {"affine_mtx.extent.2", 16},
        },
        {{0, 255},
          {0, 255},
          {0, 2},
          {0, 15}},
        options,
        dont_inline);
  }

};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::SpatialTransformerForwardGenerator, 
    spatial_transformer_forward)

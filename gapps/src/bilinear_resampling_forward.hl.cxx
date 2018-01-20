#include "algorithms/bilinear_resampling.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class BilinearResamplingForwardGenerator : 
  public Generator<BilinearResamplingForwardGenerator> {
public:
  Input<Buffer<float>>  input{"input", 4};
  Input<Buffer<float>>  warp{"warp", 4};
  Output<Buffer<float>> output{"output", 4};

  void generate() {
    Func f_output = bilinear_resampling(input, warp);
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
        {"warp.min.0", 0},
        {"warp.min.1", 0},
        {"warp.min.2", 0},
        {"warp.min.3", 0},
        {"warp.extent.0", 256},
        {"warp.extent.1", 256},
        {"warp.extent.1", 2},
        {"warp.extent.3", 16},
        },
        {{0, 255}, {0, 255}, {0, 2}, {0, 15}},
        options,
        dont_inline);
  }

};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilinearResamplingForwardGenerator, 
    bilinear_resampling_forward)

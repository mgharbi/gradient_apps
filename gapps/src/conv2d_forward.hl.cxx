#include "algorithms/conv2d_fwd.h"

namespace gradient_apps {

class Conv2dForwardGenerator : public Generator<Conv2dForwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 4};
    Input<Buffer<float>>  filter{"filter", 4};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
      std::map<std::string, Func> func_map = conv2d_fwd(
          input, filter);

      Func f_output = func_map["output"];
      output(x, y, co, n) = f_output(x, y, co, n);

      SimpleAutoscheduleOptions options;
      options.gpu = get_target().has_gpu_feature();
      std::vector<Func> funcs{output};
      simple_autoschedule(funcs,
        {
        {"input.min.0", 0},
        {"input.min.1", 0},
        {"input.min.2", 0},
        {"input.min.3", 0},
        {"input.extent.0", 256},
        {"input.extent.1", 256},
        {"input.extent.2", 32},
        {"input.extent.2", 4},
        {"filter.min.0", 0},
        {"filter.min.1", 0},
        {"filter.min.3", 0},
        {"filter.min.4", 0},
        {"filter.extent.0", 3},
        {"filter.extent.1", 3},
        {"filter.extent.3", 32},
        {"filter.extent.4", 32},
        },
        {
          {{0, 255}, {0, 255}, {0, 31}, {0, 3}},
        },
        options);

    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv2dForwardGenerator, conv2d_forward)

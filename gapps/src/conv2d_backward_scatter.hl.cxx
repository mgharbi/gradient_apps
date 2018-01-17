#include "algorithms/conv2d_bwd_scatter.h"

namespace gradient_apps {

class Conv2dBackwardScatterGenerator : public Generator<Conv2dBackwardScatterGenerator> {
public:
    Input<Buffer<float>>  d_output{"d_output", 4};
    Input<Buffer<float>>  filter{"filter", 4};
    Output<Buffer<float>> d_input{"d_input", 4};

    void generate() {
      std::map<std::string, Func> func_map = conv2d_bwd_scatter(
          d_output, filter);

      Func f_d_input = func_map["d_input"];
      d_input(x, y, co, n) = f_d_input(x, y, co, n);

      SimpleAutoscheduleOptions options;
      options.gpu = get_target().has_gpu_feature();
      std::vector<Func> funcs{d_input};
      simple_autoschedule(funcs,
        {
        {"d_output.min.0", 0},
        {"d_output.min.1", 0},
        {"d_output.min.2", 0},
        {"d_output.min.3", 0},
        {"d_output.extent.0", 256},
        {"d_output.extent.1", 256},
        {"d_output.extent.2", 32},
        {"d_output.extent.2", 4},
        {"filter.min.0", 0},
        {"filter.min.1", 0},
        {"filter.min.2", 0},
        {"filter.min.3", 0},
        {"filter.extent.0", 3},
        {"filter.extent.1", 3},
        {"filter.extent.2", 32},
        {"filter.extent.3", 32},
        },
        {
          {{0, 255}, {0, 255}, {0, 31}, {0, 3}},
        },
        options);

    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv2dBackwardScatterGenerator, conv2d_backward_scatter)

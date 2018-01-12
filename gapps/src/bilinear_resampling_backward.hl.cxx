#include "algorithms/bilinear_resampling.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class BilinearResamplingBackwardGenerator
  : public Generator<BilinearResamplingBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 4};
    Input<Buffer<float>>  warp{"warp", 4};
    Input<Buffer<float>>  d_output{"d_output", 4};

    Output<Buffer<float>>  d_input{"d_input", 4};
    Output<Buffer<float>>  d_warp{"d_warp", 4};

    void generate() {
        std::map<std::string, Func> func_map = bilinear_resampling(
            input, warp);
        Func f_output = func_map["output"];
        Func f_input = func_map["input"];
        Func f_warp = func_map["warp"];

        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}
             });
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, f_input, d_input);
        assign_gradient(adjoints, f_warp, d_warp);

        SimpleAutoscheduleOptions options;
        options.gpu = get_target().has_gpu_feature();

        std::set<std::string> dont_inline = {};

        std::vector<Func> funcs{d_input, d_warp};

        simple_autoschedule(funcs,
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
              {"warp.extent.2", 2},
              {"warp.extent.3", 16},
              {"d_output.min.0", 0},
              {"d_output.min.1", 0},
              {"d_output.min.2", 0},
              {"d_output.min.3", 0},
              {"d_output.extent.0", 256},
              {"d_output.extent.1", 256},
              {"d_output.extent.2", 3},
              {"d_output.extent.3", 16}
            },
            {
              {{0, 255}, {0, 255}, {0, 2}, {0, 15}},
              {{0, 255}, {0, 255}, {0, 1}, {0, 15}},
            },
            options,
            dont_inline);

    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilinearResamplingBackwardGenerator, 
    bilinear_resampling_backward)

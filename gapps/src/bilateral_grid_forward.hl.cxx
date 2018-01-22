#include "algorithms/bilateral_grid.h"
#include "gradient_helpers.h"

namespace gradient_apps {

class BilateralGridForwardGenerator : public Generator<BilateralGridForwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 3};       // x, y, channel
    Input<Buffer<float>>  filter_s{"filter_s", 1};
    Input<Buffer<float>>  filter_r{"filter_r", 1};

    Output<Buffer<float>> output{"output", 3};     // x, y, channel

    void generate() {
        std::map<std::string, Func> func_map = bilateral_grid(
            input, filter_s, filter_r);
        Func f_output = func_map["output"];
        output(x, y, c) = f_output(x, y, c);

        if(auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            options.gpu_tile_channel = 3;
            Func output_func = output;
            simple_autoschedule(output_func,
                                {{"input.min.0", 0},
                                 {"input.min.1", 0},
                                 {"input.min.2", 0},
                                 {"input.extent.0", 256},
                                 {"input.extent.1", 256},
                                 {"input.extent.2", 3},
                                 {"filter_s.min.0", 0},
                                 {"filter_s.extent.0", 5},
                                 {"filter_r.min.0", 0},
                                 {"filter_r.extent.0", 5}},
                                {{0, 255},
                                 {0, 255},
                                 {0, 2}},
                                options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralGridForwardGenerator, bilateral_grid_forward)

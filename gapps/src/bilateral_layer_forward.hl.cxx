#include "algorithms/bilateral_layer.h"

namespace gradient_apps {

class BilateralLayerForwardGenerator : public Generator<BilateralLayerForwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};       // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};     // x, y, z, input channel, output channel

    Output<Buffer<float>> output{"output", 4};     // x, y, channel, batch size

    void generate() {
        std::map<std::string, Func> func_map = bilateral_layer(
            input, guide, filter);
        Func f_output = func_map["output"];
        output(x, y, co, n) = f_output(x, y, co, n);

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
            {"input.extent.0", 128},
            {"input.extent.1", 128},
            {"input.extent.2", 64},
            {"input.extent.3", 4},
            {"guide.min.0", 0},
            {"guide.min.1", 0},
            {"guide.min.2", 0},
            {"guide.extent.0", 128},
            {"guide.extent.1", 128},
            {"guide.extent.2", 4},
            {"filter.min.0", 0},
            {"filter.min.1", 0},
            {"filter.min.2", 0},
            {"filter.min.3", 0},
            {"filter.min.4", 0},
            {"filter.extent.0", 3},
            {"filter.extent.1", 3},
            {"filter.extent.2", 3},
            {"filter.extent.3", 64},
            {"filter.extent.4", 64}
          },
          {{0, 127},
            {0, 127},
            {0, 63},
            {0, 3}},
          options,
          dont_inline);

        // if (get_target().has_gpu_feature()) {
        //   Var xi, yi;
        //   func_map["grid"]
        //     .compute_root()
        //     .gpu_tile(x, y, xi, yi, 8, 8);
        //     ;
        //   func_map["grid"]
        //     .update(0)
        //     .gpu_tile(x, y, xi, yi, 8, 8);
        //     ;
        //   func_map["grid"]
        //     .update(1)
        //     .gpu_tile(x, y, xi, yi, 8, 8);
        //     ;
        //   func_map["conv"]
        //     .compute_root()
        //     .gpu_tile(x, y, xi, yi, 8, 8);
        //     ;
        //   func_map["conv"]
        //     .update(0)
        //     .gpu_tile(x, y, xi, yi, 8, 8);
        //     ;
        // } else {
        //   func_map["grid"]
        //     .compute_root()
        //     .parallel(n)
        //     .parallel(ci)
        //     .parallel(z)
        //     .vectorize(x, 8)
        //     ;
        //   func_map["grid"]
        //     .update(0)
        //     .parallel(n)
        //     .parallel(ci)
        //     .parallel(y)
        //     .vectorize(x, 8)
        //     ;
        //   func_map["grid"]
        //     .update(1)
        //     .parallel(n)
        //     .parallel(ci)
        //     .parallel(y)
        //     .vectorize(x, 8)
        //     ;
        //   func_map["conv"]
        //     .compute_root()
        //     .parallel(n)
        //     .parallel(co)
        //     .parallel(z)
        //     .vectorize(x, 8)
        //     ;
        //   func_map["conv"]
        //     .update(0)
        //     .parallel(n)
        //     .parallel(co)
        //     .parallel(z)
        //     .vectorize(x, 8)
        //     ;
        //   output
        //     .compute_root()
        //     .parallel(n)
        //     .parallel(co)
        //     .parallel(y)
        //     .vectorize(x, 8)
        //     ;
        // }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralLayerForwardGenerator, bilateral_layer_forward)

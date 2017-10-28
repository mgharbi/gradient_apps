#include "bilateral_layer_algorithm.h"

namespace gradient_apps {

class BilateralLayerForwardGenerator : public Generator<BilateralLayerForwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};       // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};     // x, y, z offset, input channel, output channel
    Input<Buffer<float>>  bias{"bias", 2};         // z offset, channel

    Output<Buffer<float>> output{"output", 4};     // x, y, channel, batch size

    void generate() {
        std::map<std::string, Func> func_map = bilateral_layer(input, guide, filter, bias);
        Func f_output = func_map["output"];
        output(x, y, co, n) = f_output(x, y, co, n);
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralLayerForwardGenerator, bilateral_layer_forward)

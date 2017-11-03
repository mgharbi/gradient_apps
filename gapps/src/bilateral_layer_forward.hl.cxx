#include "bilateral_layer_algorithm.h"

namespace gradient_apps {

class BilateralLayerForwardGenerator : public Generator<BilateralLayerForwardGenerator> {
public:
    Input<int> sigma_x{"sigma_x"}; // block_size in x
    Input<int> sigma_y{"sigma_y"}; // block_size in y
    Input<int> sigma_z{"sigma_z"}; // number of guide discrete levels

    Input<Buffer<float>>  input{"input", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};       // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};     // x, y, z, input channel, output channel

    Output<Buffer<float>> output{"output", 4};     // x, y, channel, batch size

    void generate() {
        std::map<std::string, Func> func_map = bilateral_layer(
            input, guide, filter, sigma_x, sigma_y, sigma_z);
        Func f_output = func_map["output"];
        output(x, y, co, n) = f_output(x, y, co, n);

        func_map["grid"].compute_root();
        func_map["conv"].compute_root();
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralLayerForwardGenerator, bilateral_layer_forward)

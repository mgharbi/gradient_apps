#include "bilateral_layer_algorithm.h"

namespace gradient_apps {

class BilateralLayerBackwardGenerator : public Generator<BilateralLayerBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};       // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};     // x, y, z offset, input channel, output channel
    Input<Buffer<float>>  bias{"bias", 2};         // z offset, channel
    Input<Buffer<float>>  adjoint{"adjoint", 4};   // x, y, channel, batch size

    Output<Buffer<float>> d_input{"d_input", 4};   // same as input
    Output<Buffer<float>> d_guide{"d_guide", 3};   // same as guide
    Output<Buffer<float>> d_filter{"d_filter", 5}; // same as filter
    Output<Buffer<float>> d_bias{"d_bias", 2};     // same as bias

    void generate() {
        std::map<std::string, Func> func_map = bilateral_layer(input, guide, filter, bias);
        Func f_output = func_map["output"];
        Func f_input = func_map["input"];
        Func f_guide = func_map["guide"];
        Func f_filter = func_map["filter"];
        Func f_bias = func_map["bias"];
        
        Derivative d = propagate_adjoints(f_output, adjoint,
                                          {{adjoint.dim(0).min(), adjoint.dim(0).max()},
                                           {adjoint.dim(1).min(), adjoint.dim(1).max()},
                                           {adjoint.dim(2).min(), adjoint.dim(2).max()},
                                           {adjoint.dim(3).min(), adjoint.dim(3).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        Func f_d_input = adjoints[FuncKey{f_input.name(), -1}];
        Func f_d_guide = adjoints[FuncKey{f_guide.name(), -1}];
        Func f_d_filter = adjoints[FuncKey{f_filter.name(), -1}];
        Func f_d_bias = adjoints[FuncKey{f_bias.name(), -1}];

        d_input(x, y, ci, n) = f_d_input(x, y, ci, n);
        d_guide(x, y, n) = f_d_guide(x, y, n);
        d_filter(x, y, z, ci, co) = f_d_filter(x, y, z, ci, co);
        d_bias(z, ci) = f_d_bias(z, ci);
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralLayerBackwardGenerator, bilateral_layer_backward)

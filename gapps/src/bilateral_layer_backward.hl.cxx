#include "bilateral_layer_algorithm.h"

namespace gradient_apps {

void apply_compute_root(Func F) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
    }
}

class BilateralLayerBackwardGenerator : public Generator<BilateralLayerBackwardGenerator> {
public:
    Input<int> sigma_x{"sigma_x"}; // block_size in x
    Input<int> sigma_y{"sigma_y"}; // block_size in y
    Input<int> sigma_z{"sigma_z"}; // number of guide discrete levels

    Input<Buffer<float>>  input{"input", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};       // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};     // x, y, z offset, input channel, output channel

    Input<Buffer<float>>  d_output{"d_output", 4};   // x, y, out_channel, batch size

    Output<Buffer<float>> d_input{"d_input", 4};   // same as input
    Output<Buffer<float>> d_guide{"d_guide", 3};   // same as guide
    Output<Buffer<float>> d_filter{"d_filter", 5}; // same as filter

    void generate() {
        std::map<std::string, Func> func_map = bilateral_layer(
            input, guide, filter, sigma_x, sigma_y, sigma_z);

        Func f_output = func_map["output"];
        Func f_input = func_map["input"];
        Func f_guide = func_map["guide"];
        Func f_filter = func_map["filter"];
        
        Derivative d = propagate_adjoints(f_output, d_output,
                                          {{d_output.dim(0).min(), d_output.dim(0).max()},
                                           {d_output.dim(1).min(), d_output.dim(1).max()},
                                           {d_output.dim(2).min(), d_output.dim(2).max()},
                                           {d_output.dim(3).min(), d_output.dim(3).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_input.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_guide.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_filter.name(), -1}) != adjoints.end());
        Func f_d_input  = adjoints[FuncKey{f_input.name(), -1}];
        Func f_d_guide  = adjoints[FuncKey{f_guide.name(), -1}];
        Func f_d_filter = adjoints[FuncKey{f_filter.name(), -1}];

        apply_compute_root(f_d_input);
        apply_compute_root(f_d_guide);
        apply_compute_root(f_d_filter);

        d_input(x, y, ci, n) = f_d_input(x, y, ci, n);
        d_guide(x, y, n) = f_d_guide(x, y, n);
        d_filter(x, y, z, ci, co) = f_d_filter(x, y, z, ci, co);
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralLayerBackwardGenerator, bilateral_layer_backward)

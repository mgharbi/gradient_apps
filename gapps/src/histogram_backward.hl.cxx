#include "algorithms/histogram.h"

namespace gradient_apps {

class HistogramBackwardGenerator : public Generator<HistogramBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 2};
    Input<Buffer<float>>  d_output{"d_output", 1};
    Input<int> nbins{"nbins"};
    Output<Buffer<float>> d_input{"d_input", 2};

    void generate() {
        std::map<std::string, Func> func_map = histogram(input, nbins);
        Func f_input = func_map["input"];
        Func f_output = func_map["output"];

        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()}});

        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_input.name(), -1}) != adjoints.end());

        Func f_d_input  = adjoints[FuncKey{f_input.name(), -1}];
        d_input(x, y) = f_d_input(x, y);

        if(auto_schedule) {
          printf("Autoscheduling forward\n");
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::HistogramBackwardGenerator, histogram_backward)

#include "algorithms/naive_demosaick.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class NaiveDemosaickBackwardGenerator : public Generator<NaiveDemosaickBackwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 2};
    Input<Buffer<float>>  d_output{"d_output", 3};
    Output<Buffer<float>> d_mosaick{"d_mosaick", 2};

    void generate() {
        std::map<std::string, Func> func_map = naive_demosaick(mosaick);
        Func f_output = func_map["output"];
        Func f_mosaick = func_map["mosaick"];

        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_mosaick.name(), -1}) != adjoints.end());

        Func f_d_mosaick  = adjoints[FuncKey{f_mosaick.name(), -1}];

        d_mosaick(x, y) = f_d_mosaick(x, y);

        if(auto_schedule) {
        } else {
          compute_all_root(d_mosaick);
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::NaiveDemosaickBackwardGenerator, naive_demosaick_backward)

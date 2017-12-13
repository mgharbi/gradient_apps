#include "algorithms/learnable_demosaick.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class LearnableDemosaickBackwardGenerator 
  : public Generator<LearnableDemosaickBackwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 3};
    Input<Buffer<float>>  sel_filts{"selection_filters", 3};
    Input<Buffer<float>>  green_filts{"green_filters", 3};
    Input<Buffer<float>>  d_output{"d_output", 4};

    Output<Buffer<float>> d_mosaick{"d_mosaick", 3};
    Output<Buffer<float>> d_sel_filts{"d_selection_filters", 3};
    Output<Buffer<float>> d_green_filts{"d_green_filters", 3};

    void generate() {
        std::map<std::string, Func> func_map = learnable_demosaick(mosaick, sel_filts, green_filts);
        Func f_output = func_map["output"];
        Func f_mosaick = func_map["mosaick"];
        Func f_sel_filts = func_map["selection_filters"];
        Func f_green_filts = func_map["green_filters"];

        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}
             });
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_mosaick.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_sel_filts.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_green_filts.name(), -1}) != adjoints.end());

        Func f_d_mosaick  = adjoints[FuncKey{f_mosaick.name(), -1}];
        Func f_d_sel_filts  = adjoints[FuncKey{f_sel_filts.name(), -1}];
        Func f_d_green_filts  = adjoints[FuncKey{f_green_filts.name(), -1}];

        d_mosaick(x, y, n) = f_d_mosaick(x, y, n);
        d_sel_filts(x, y, n) = f_d_sel_filts(x, y, n);
        d_green_filts(x, y, n) = f_d_green_filts(x, y, n);

        Var xi("xi"), yi("yi"), xy("xy"), xyn("xyn");
        compute_all_root(d_sel_filts);
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::LearnableDemosaickBackwardGenerator, learnable_demosaick_backward)

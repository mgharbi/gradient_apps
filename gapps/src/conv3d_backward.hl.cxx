#include "algorithms/conv3d.h"
#include <iostream>

#include "gradient_helpers.h"

using std::cout;
using std::endl;

namespace gradient_apps {


class Conv3dBackwardGenerator : public Generator<Conv3dBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 5};
    Input<Buffer<float>>  filter{"filter", 5};
    Input<Buffer<float>>  d_output{"d_output", 5};

    Output<Buffer<float>> d_input{"d_input", 5};
    Output<Buffer<float>> d_filter{"d_filter", 5};

    void generate() {
        std::map<std::string, Func> func_map = conv3d(
            input, filter);

        Func f_input = func_map["input"];
        Func f_filter = func_map["filter"];
        Func f_output = func_map["output"];
        
        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()},
             {d_output.dim(4).min(), d_output.dim(4).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_input.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_filter.name(), -1}) != adjoints.end());

        Func f_d_input  = adjoints[FuncKey{f_input.name(), -1}];
        Func f_d_filter = adjoints[FuncKey{f_filter.name(), -1}];

        d_input(x, y, z, ci, n) = f_d_input(x, y, z, ci, n);
        d_filter(x, y, z, ci, co) = f_d_filter(x, y, z, ci, co);

        if(auto_schedule) {
        } else {
          // Forward schedule -------------------------------------------------
          f_output
            .compute_root()
            .parallel(n)
            .parallel(co)
            .parallel(z)
            .vectorize(x, 8)
            ;

          printf("\nd_input deps:\n\n");
          print_func(d_input);
          printf("\nd_filter deps:\n\n");
          print_func(d_filter);

          // Backward schedule -------------------------------------------------
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv3dBackwardGenerator, conv3d_backward)

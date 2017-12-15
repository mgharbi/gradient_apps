#include "algorithms/conv1d.h"
#include <iostream>

#include "gradient_helpers.h"

using std::cout;
using std::endl;

namespace gradient_apps {


class Conv1dBackwardGenerator : public Generator<Conv1dBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 3};
    Input<Buffer<float>>  filter{"filter", 3};
    Input<Buffer<float>>  d_output{"d_output", 3};

    Output<Buffer<float>> d_input{"d_input", 3};
    // Output<Buffer<float>> d_filter{"d_filter", 3};

    void generate() {
        std::map<std::string, Func> func_map = conv1d(
            input, filter);

        Func f_input = func_map["input"];
        Func f_filter = func_map["filter"];
        Func f_output = func_map["output"];
        
        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_input.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_filter.name(), -1}) != adjoints.end());

        Func f_d_input  = adjoints[FuncKey{f_input.name(), -1}];
        // Func f_d_filter = adjoints[FuncKey{f_filter.name(), -1}];

        d_input(x, ci, n) = f_d_input(x, ci, n);
        // d_filter(x, ci, co) = f_d_filter(x, ci, co);

        if(auto_schedule) {
        } else {
          Var nc("nc");
          // Forward schedule -----------------------------------

          // Backward schedule -----------------------------------
          printf("\nd_input deps:\n\n");
          print_func(d_input);
          // printf("\nd_filter deps:\n\n");
          // print_func(d_filter);

          auto flist_input = get_deps(d_input);

          // This just does a copy of the repeat_edge, constant_exterior
          Func d_input(flist_input["d_input"]);
          d_input
            .compute_root()
            .fuse(n, ci, nc)
            .parallel(nc)
            .vectorize(x, 8);
            ;

          Func f_input_0_d_def__(flist_input["f_input_0_d_def__"]);
          f_input_0_d_def__
            .compute_at(d_input, x)
            .vectorize(x, 8);
            ;
          f_input_0_d_def__
            .update()
            .vectorize(x, 8)
            ;
          // f_input_0_d_def__
          //   .compute_root()
          //   .fuse(n, ci, nc)
          //   .parallel(nc)
          //   .vectorize(x, 8);
          //   ;
          // f_input_0_d_def__
          //   .update()
          //   .fuse(n, ci, nc)
          //   .parallel(nc)
          //   .vectorize(x, 8);
          //   ;

          // Func f_output_1_d__(flist_input["f_output_1_d__"]);
          // f_output_1_d__
          //   .compute_root()
          //   .fuse(n, co, nc)
          //   .parallel(nc)
          //   .vectorize(x, 8)
          //   ;

          // auto flist_filter = get_deps(d_filter);
          // Func f_filter_0_d_def__(flist_filter["f_filter_0_d_def__"]);
          // f_filter_0_d_def__
          //   .compute_root()
          //   .fuse(co, ci, nc)
          //   .parallel(nc)
          //   .vectorize(x, 2)
          //   ;
          //
          // rvars = f_filter_0_d_def__.rvars();  // ci, x, y, z
          // Var r0("r0");
          // Var r1("r1");
          // Var r2("r2");
          // Var r3("r3");
          // Func intrm = f_filter_0_d_def__
          //   .update()
          //   .rfactor({{rvars[3], r3}});
          //
          // intrm
          //   .compute_at(f_filter_0_d_def__, x)
          //   .fuse(co, ci, nc)
          //   .parallel(nc)
          //   .vectorize(x, 2)
          //   ;
          // intrm
          //   .update()
          //   .parallel(r3);

          // Backward schedule -------------------------------------------------
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv1dBackwardGenerator, conv1d_backward)

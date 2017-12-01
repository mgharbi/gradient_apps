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
          // Forward schedule -----------------------------------
          Var nc("nc");
          Var ncz("ncz");
          Var nczy("nczy");
          f_output
            .compute_root()
            .fuse(n, co, nc)
            .fuse(nc, z, ncz)
            .fuse(ncz, y, nczy)
            .parallel(nczy)
            .vectorize(x, 8)
            ;

          // Backward schedule -----------------------------------
          printf("\nd_input deps:\n\n");
          print_func(d_input);
          printf("\nd_filter deps:\n\n");
          print_func(d_filter);

          auto flist_input = get_deps(d_input);

          // Func f_output_1_d__(flist_input["f_output_1_d__"]);
          // f_output_1_d__
          //   .compute_root()
          //   .fuse(n, co, nc)
          //   .fuse(nc, z, ncz)
          //   .fuse(ncz, y, nczy)
          //   .parallel(nczy)
          //   .vectorize(x, 8)
          //   ;

          // This just does a copy of the repeat_edge, constant_exterior
          Func d_input(flist_input["d_input"]);
          d_input
            .compute_root()
            .fuse(n, ci, nc)
            .fuse(nc, z, ncz)
            .fuse(ncz, y, nczy)
            .parallel(nczy)
            .vectorize(x, 8);
            ;

          Func f_input_0_d_def__(flist_input["f_input_0_d_def__"]);
          f_input_0_d_def__
            .compute_root()
            .fuse(n, ci, nc)
            .fuse(nc, z, ncz)
            .fuse(ncz, y, nczy)
            .parallel(nczy)
            .vectorize(x, 8);
            ;

          std::vector<RVar> rvars = f_input_0_d_def__.rvars(0);
          for(RVar r: rvars) {
            cout << "rvar " << r.name() << "\n";
          }
          f_input_0_d_def__
            .update()
            // .reorder(rvars[1], rvars[2], rvars[3], rvars[0])
            .fuse(n, ci, nc)
            .fuse(nc, z, ncz)
            .fuse(ncz, y, nczy)
            .parallel(nczy)
            .vectorize(x, 8)
            ;

          auto flist_filter = get_deps(d_filter);
          Func f_filter_0_d_def__(flist_filter["f_filter_0_d_def__"]);
          f_filter_0_d_def__
            .compute_root()
            .fuse(co, ci, nc)
            .fuse(nc, z, ncz)
            .fuse(ncz, y, nczy)
            .parallel(nczy)
            .vectorize(x, 2)
            ;

          rvars = f_filter_0_d_def__.rvars();  // ci, x, y, z
          Var r0("r0");
          Var r1("r1");
          Var r2("r2");
          Var r3("r3");
          Func intrm = f_filter_0_d_def__
            .update()
            .rfactor({{rvars[3], r3}});

          intrm
            .compute_at(f_filter_0_d_def__, x)
            .fuse(co, ci, nc)
            .fuse(nc, z, ncz)
            .fuse(ncz, y, nczy)
            .parallel(nczy)
            .vectorize(x, 2)
            ;
          intrm
            .update()
            .parallel(r3);

          // Backward schedule -------------------------------------------------
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv3dBackwardGenerator, conv3d_backward)

#include "algorithms/conv1d.h"
#include <iostream>

#include "gradient_helpers.h"

using std::cout;
using std::endl;

namespace gradient_apps {


class Conv1dManualBackwardGenerator : public Generator<Conv1dManualBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 3};
    Input<Buffer<float>>  filter{"filter", 3};
    Input<Buffer<float>>  d_output{"d_output", 3};
    Output<Buffer<float>> d_input{"d_input", 3};

    void generate() {
      bool use_ce = false;
      Func f_filter("f_filter");
      f_filter(x, ci, co) = filter(x, ci, co);

      Expr width = d_output.dim(0).extent();
      Expr kw = filter.dim(0).extent();
      Expr out_chans = d_output.dim(1).extent();
      Expr in_chans = input.dim(1).extent();
      Expr bsize = input.dim(2).extent();


      if(true) {
        Func f_output_d("f_output_d");
        f_output_d(x, co, n) = d_output(x, co, n);
        // f_output_d(x, co, n) = Halide::BoundaryConditions::constant_exterior(
        //     d_output, 0.0f)(x, co, n);
        
        // f_output_d(rfirst.x, rfirst.y, rfirst.z) = d_output(rfirst.x, rfirst.y, rfirst.z);
        // select(0 <= x && x < width,
        //     d_output(x, co, n), 0.0f);

        Func f_input_d("f_input_d");
        RDom r(0, kw, 0, out_chans);
        r.where((x + r.x) < width);
        f_input_d(x, ci, n) = 0.0f;
        f_input_d(x, ci, n) = f_input_d(x, ci, n) + 
          f_output_d(x + r.x, r.y, n)*f_filter(r.x, ci, r.y);

        d_input(x, ci, n) = f_input_d(x, ci, n);

        Var nc("nc");

        d_input
          .compute_root()
          .fuse(n, ci, nc)
          .parallel(nc)
          .vectorize(x, 8)
          ;
        f_input_d
          .compute_at(d_input, x)
          .vectorize(x, 8);
        f_input_d
          .update()
          .vectorize(x, 8);

      } else {
        // Func f_output_1_d_def("f_output_1_d_def");
        // f_output_1_d_def(x, co, n) = d_output(x, co, n);
        //
        // Func re1("re1");
        // Func ce("ce");
        // Func f_output_1_d("f_output_1_d");
        // if(use_ce) {
        //   re1(x, co, n) = Halide::BoundaryConditions::repeat_edge(f_output_1_d_def,
        //     {{0, width}, {0, in_chans}, {0, bsize}})(x, co, n);
        //   ce(x, co, n) = Halide::BoundaryConditions::constant_exterior(re1, 0.0f,
        //     {{0, width}, {0, in_chans}, {0, bsize}})(x, co, n);
        //   f_output_1_d(x, co, n) = ce(x, co, n);
        // } else {
        //   f_output_1_d(x, co, n) = f_output_1_d_def(x, co, n);
        // }
        //
        // Func f_input_0_d_def("f_input_0_d_def");
        // RDom r(0, kw, 0, out_chans);
        // r.where((x + kw/2 - r.x) < d_output.dim(0).extent() + d_output.dim(0).extent());
        // r.where((x + kw/2 - r.x) >= d_output.dim(0).min());
        // f_input_0_d_def(x, ci, n) = 0.0f;
        // f_input_0_d_def(x, ci, n) = f_input_0_d_def(x, ci, n) + 
        //   f_output_1_d(x + kw/2 - r.x, r.y, n)*f_filter(r.x, ci, r.y);
        //
        // Func re2("re2");
        // Func ce1("ce1");
        // Func f_input_0_d("f_input_0_d");
        // if (use_ce) {
        //   re2(x, ci, n) = Halide::BoundaryConditions::repeat_edge(f_input_0_d_def,
        //       {{0, width}, {0, in_chans}, {0, bsize}})(x, ci, n);
        //   ce1(x, ci, n) = Halide::BoundaryConditions::constant_exterior(re2, 0.0f,
        //       {{0, width}, {0, in_chans}, {0, bsize}})(x, ci, n);
        //   f_input_0_d(x, ci, n) = ce1(x, ci, n);
        //   d_input(x, ci, n) = f_input_0_d(x, ci, n);
        // } else {
        //   d_input(x, ci, n) = f_input_0_d_def(x, ci, n);
        // }
        //
        // Var nc("nc");
        //
        // // ce.compute_root().fuse(n, co, nc).parallel(nc).vectorize(x, 8);
        // // ce1.compute_root().fuse(n, ci, nc).parallel(nc).vectorize(x, 8);
        //
        // d_input
        //   .compute_root()
        //   .fuse(n, ci, nc)
        //   .parallel(nc)
        //   .vectorize(x, 8)
        //   ;
        // f_input_0_d_def
        //   .compute_at(d_input, x)
        //   .vectorize(x, 8);
        // f_input_0_d_def
        //   .update()
        //   .vectorize(x, 8);
      }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv1dManualBackwardGenerator, conv1d_manual_backward)

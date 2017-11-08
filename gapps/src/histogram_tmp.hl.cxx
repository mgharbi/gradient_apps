#include "algorithms/histogram.h"

namespace gradient_apps {

class HistogramTMPGenerator : public Generator<HistogramTMPGenerator> {
public:
    Input<Buffer<float>>  input{"input", 2};
    Input<Buffer<float>>  d_output{"d_output", 1};
    Input<int> nbins{"nbins"};
    Output<Buffer<float>> d_input{"d_input", 2};

    void generate() {

      Func f_input_0_d_def_("f_input_0_d_def_");

      RDom r(0, input.dim(0).extent(), 0, input.dim(1).extent());

      Expr t4 = input(r.x, r.y);
      f_input_0_d_def_(x, y) = 0.0f;
      f_input_0_d_def_(r.x, r.y) = f_input_0_d_def_(r.x, r.y) + 
        select(((1.0f < t4) || (t4 < 0.0f)), 0.0f, 
            ((0.0f - d_output(
              max(cast<int>(floor((max(min(t4, 1.0f), 0.0f)*cast<float>((nbins + -1))))), 0))
             )*cast<float>((nbins-1))));
      d_input(x, y) = f_input_0_d_def_(x, y);
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::HistogramTMPGenerator, histogram_tmp)

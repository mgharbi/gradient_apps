#include "algorithms/learnable_demosaick.h"

namespace gradient_apps {

class LearnableDemosaickForwardGenerator : public Generator<LearnableDemosaickForwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 3};
    Input<Buffer<float>>  gfilt{"gfilt", 1};
    Input<Buffer<float>>  grad_filt{"grad_filt", 1};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        std::map<std::string, Func> func_map = learnable_demosaick(mosaick, gfilt, grad_filt);
        Func f_output = func_map["output"];
        Func dx = func_map["dx"];
        Func dy = func_map["dy"];
        Func v_interp_g = func_map["v_interp_g"];
        Func h_interp_g = func_map["h_interp_g"];
        output(x, y, c, n) = f_output(x, y, c, n);

        if(auto_schedule) {
        } else {
          Var xi("xi"), yi("yi"), xy("xy");
          // dx
          //   .compute_at(output, xy)
          //   .vectorize(x, 8)
          //   ;
          // dy
          //   .compute_at(output, xy)
          //   .vectorize(x, 8)
          //   ;
          // v_interp_g
          //   .compute_at(output, xy)
          //   .vectorize(x, 8)
          //   ;
          // h_interp_g
          //   .compute_at(output, xy)
          //   .vectorize(x, 8)
          //   ;

          output
            .tile(x, y, xi, yi, 16, 16)
            .fuse(x, y, xy)
            .parallel(xy, 8)
            .vectorize(xi, 8)
            ;
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::LearnableDemosaickForwardGenerator, learnable_demosaick_forward)

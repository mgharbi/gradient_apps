#include "algorithms/naive_demosaick.h"

namespace gradient_apps {

class NaiveDemosaickForwardGenerator : public Generator<NaiveDemosaickForwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 3};       // x, y
    Output<Buffer<float>> output{"output", 4};     // x, y, 3


    void generate() {
        std::map<std::string, Func> func_map = naive_demosaick(mosaick);
        Func f_output = func_map["output"];
        output(x, y, c, n) = f_output(x, y, c, n);

        if(auto_schedule) {
        } else {
          Var xi("xi"), yi("yi"), xy("xy"), xyn("xyn");
            output
              .tile(x, y, xi, yi, 16, 16)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .parallel(xyn, 8)
              .vectorize(xi, 8)
              ;
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::NaiveDemosaickForwardGenerator, naive_demosaick_forward)

#include "algorithms/naive_demosaick.h"

namespace gradient_apps {

class NaiveDemosaickForwardGenerator : public Generator<NaiveDemosaickForwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 2};       // x, y
    Output<Buffer<float>> output{"output", 3};     // x, y, 3


    void generate() {
        std::map<std::string, Func> func_map = naive_demosaick(mosaick);
        Func f_output = func_map["output"];
        output(x, y, c) = f_output(x, y, c);

        if(auto_schedule) {
        } else {
          // Var yc("yc");
          Var xi("xi"), yi("yi"), xy("xy");
          output
            // .fuse(y, c, yc)
            // .parallel(yc, 8)
            .tile(x, y, xi, yi, 8, 8)
            .fuse(x, y, xy)
            .parallel(xy)
            .vectorize(xi, 8)
            ;
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::NaiveDemosaickForwardGenerator, naive_demosaick_forward)

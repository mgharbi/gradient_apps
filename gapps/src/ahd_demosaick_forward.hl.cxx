#include "algorithms/ahd_demosaick.h"

namespace gradient_apps {

class AHDForwardGenerator : public Generator<AHDForwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 2};       // x, y
    Output<Buffer<float>> output{"output", 3};     // x, y, 3


    void generate() {
        std::map<std::string, Func> func_map = ahd_demosaick(mosaick);
        Func f_output = func_map["output"];
        output(x, y, c) = f_output(x, y, c);
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::AHDForwardGenerator, ahd_demosaick_forward)

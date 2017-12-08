#include "algorithms/learnable_wiener.h"

namespace gradient_apps {

class LearnableWienerForwardGenerator
  : public Generator<LearnableWienerForwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  reg_kernel{"reg_kernel", 2};
    Output<Buffer<float>> output{"output", 3};

    void generate() {
        std::map<std::string, Func> func_map =
            learnable_wiener(blurred, kernel, reg_kernel);
        Func filtered = func_map["filtered"];
        output(x, y, c) = filtered(x, y, c);
        // filtered.estimate(x, 0, blurred.width())
        //         .estimate(y, 0, blurred.height())
        //         .estimate(c, 0, blurred.channels());
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::LearnableWienerForwardGenerator, learnable_wiener_forward)

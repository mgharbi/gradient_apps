#include "gradient_helpers.h"

#include "algorithms/deconv_cg_weight.h"

namespace gradient_apps {

class DeconvCgWeightForwardGenerator
  : public Generator<DeconvCgWeightForwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  current{"current", 3};
    Input<Buffer<float>>  reg_kernels{"reg_kernel", 3};
    Input<Buffer<float>>  reg_target_kernels{"reg_target_kernels", 3};
    Input<Buffer<float>>  reg_powers{"reg_powers", 1};
    Output<Buffer<float>> weights{"weights", 4};

    void generate() {
        auto func_map = deconv_cg_weight(blurred, current,
            reg_kernels, reg_target_kernels, reg_powers);
        weights(x, y, c, n) = func_map["weights"](x, y, c, n);

        if (auto_schedule) {
        } else {
            auto func_map = get_deps(weights);
            compute_all_root(weights);
            Func rKc = Func(func_map["rKc"]);
            rKc.update()
               .parallel(y)
               .vectorize(x, 16);
            Func rtKb = Func(func_map["rtKb"]);
            rtKb.update()
                .parallel(y)
                .vectorize(x, 16);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgWeightForwardGenerator, deconv_cg_weight_forward)

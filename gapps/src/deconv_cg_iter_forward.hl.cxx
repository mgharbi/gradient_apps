#include "gradient_helpers.h"

#include "algorithms/deconv_cg_iter.h"

namespace gradient_apps {

class DeconvCgIterForwardGenerator
  : public Generator<DeconvCgIterForwardGenerator> {
public:
    Input<Buffer<float>>  xrp{"xrp", 4};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernel", 3};
    Output<Buffer<float>> next_xrp{"next_xrp", 4};

    void generate() {
        auto func_map = deconv_cg_iter(xrp, kernel, reg_kernel_weights, reg_kernels);
        next_xrp(x, y, c, n) = func_map["next_xrp"](x, y, c, n);

        if (auto_schedule) {
            xrp.dim(0).set_bounds_estimate(0, 320);
            xrp.dim(1).set_bounds_estimate(0, 240);
            xrp.dim(2).set_bounds_estimate(0, 3);
            xrp.dim(3).set_bounds_estimate(0, 3);

            kernel.dim(0).set_bounds_estimate(0, 5);
            kernel.dim(1).set_bounds_estimate(0, 5);

            reg_kernel_weights.dim(0).set_bounds_estimate(0, 2);

            reg_kernels.dim(0).set_bounds_estimate(0, 3);
            reg_kernels.dim(1).set_bounds_estimate(0, 3);
            reg_kernels.dim(2).set_bounds_estimate(0, 2);

            next_xrp.estimate(x, 0, 320)
                    .estimate(y, 0, 240)
                    .estimate(c, 0, 3)
                    .estimate(n, 0, 3);
        } else {
            compute_all_root(next_xrp);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgIterForwardGenerator, deconv_cg_iter_forward)

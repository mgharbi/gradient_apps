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
    Input<Buffer<float>>  precond_kernel{"precond_kernel", 2};
    Input<Buffer<float>>  w_kernel{"w_kernel", 3};
    Input<Buffer<float>>  w_reg_kernels{"w_reg_kernels", 4};
    Output<Buffer<float>> next_xrp{"next_xrp", 4};

    void generate() {
        auto func_map = deconv_cg_iter(xrp, kernel,
            reg_kernel_weights, reg_kernels,
            precond_kernel, w_kernel, w_reg_kernels);
        next_xrp(x, y, c, n) = func_map["next_xrp"](x, y, c, n);

        if (auto_schedule) {
            xrp.dim(0).set_bounds_estimate(0, 320);
            xrp.dim(1).set_bounds_estimate(0, 240);
            xrp.dim(2).set_bounds_estimate(0, 3);
            xrp.dim(3).set_bounds_estimate(0, 3);

            kernel.dim(0).set_bounds_estimate(0, 7);
            kernel.dim(1).set_bounds_estimate(0, 7);

            reg_kernel_weights.dim(0).set_bounds_estimate(0, 2);

            reg_kernels.dim(0).set_bounds_estimate(0, 3);
            reg_kernels.dim(1).set_bounds_estimate(0, 3);
            reg_kernels.dim(2).set_bounds_estimate(0, 2);

            next_xrp.estimate(x, 0, 320)
                    .estimate(y, 0, 240)
                    .estimate(c, 0, 3)
                    .estimate(n, 0, 3);
        } else {
            auto func_map = get_deps(next_xrp);
            compute_all_root(next_xrp);
            Func Kp = Func(func_map["Kp"]);
            Kp.update()
              .parallel(y)
              .vectorize(x, 16);
            Func KTWKp = Func(func_map["K^TWKp"]);
            KTWKp.update()
                 .parallel(y)
                 .vectorize(x, 16);
            Func rKp = Func(func_map["rKp"]);
            rKp.update()
               .parallel(y)
               .vectorize(x, 16);
            Func rKTWrKp = Func(func_map["rK^TWrKp"]);
            rKTWrKp.update()
                   .parallel(y)
                   .vectorize(x, 16);
            Func Pr = Func(func_map["Pr"]);
            Pr.update()
              .parallel(y)
              .vectorize(x, 16);
            Func next_z = Func(func_map["next_z"]);
            next_z.update()
                  .parallel(y)
                  .vectorize(x, 16);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgIterForwardGenerator, deconv_cg_iter_forward)

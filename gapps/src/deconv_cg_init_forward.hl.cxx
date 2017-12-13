#include "gradient_helpers.h"

namespace gradient_apps {

class DeconvCgInitGenerator
  : public Generator<DeconvCgInitGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  x0{"x0", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernel", 3};
    Output<Buffer<float>> xrp{"xrp", 4};

    void generate() {
        // Initializing conjugate gradient
        // Want to solve A^TAx = A^Tb
        // A -> correlation with kernel
        // A^T -> convolution with kernel
        // Initializing r0 = A^Tb - A^TAx0
        Var x("x"), y("y"), c("c"), n("n");
        RDom r_kernel(kernel);
        Func clamped_b = BoundaryConditions::repeat_edge(blurred);
        Func clamped_x0 = BoundaryConditions::repeat_edge(x0);
        Func KTb("K^Tb");
        KTb(x, y, c) = 0.f;
        KTb(x, y, c) += clamped_b(x + r_kernel.x - kernel.width()  / 2,
                                  y + r_kernel.y - kernel.height() / 2,
                                  c) *
                        kernel(kernel.width()  - r_kernel.x - 1,
                               kernel.height() - r_kernel.y - 1);
        RDom r_reg_kernel(reg_kernels);
        Func rKTb("rK^Tb");
        rKTb(x, y, c, n) = 0.f;
        rKTb(x, y, c, n) += clamped_b(x + r_reg_kernel.x - reg_kernels.width()  / 2,
                                      y + r_reg_kernel.y - reg_kernels.height() / 2,
                                      c) *
                            reg_kernels(reg_kernels.width()  - r_reg_kernel.x - 1,
                                        reg_kernels.height() - r_reg_kernel.y - 1,
                                        n);
        Func ATb("A^Tb");
        ATb(x, y, c) = 0.f;
        ATb(x, y, c) += KTb(x, y, c);
        ATb(x, y, c) += rKTb(x, y, c, r_reg_kernel.z) * reg_kernel_weights(r_reg_kernel.z);

        Func Kx0("Kx0");
        Kx0(x, y, c)  = 0.f;
        Kx0(x, y, c) += clamped_x0(x + r_kernel.x - kernel.width()  / 2,
                                   y + r_kernel.y - kernel.height() / 2,
                                   c) *
                        kernel(r_kernel.x, r_kernel.y);
        Func KTKx0("K^TKx0");
        KTKx0(x, y, c)  = 0.f;
        KTKx0(x, y, c) += Kx0(x + r_kernel.x - kernel.width()  / 2,
                              y + r_kernel.y - kernel.height() / 2,
                              c) *
                          kernel(kernel.width()  - r_kernel.x - 1,
                                 kernel.height() - r_kernel.y - 1);
        Func rKx0("rKx0");
        rKx0(x, y, c, n) = 0.f;
        rKx0(x, y, c, n) += clamped_x0(x + r_reg_kernel.x - reg_kernels.width()  / 2,
                                       y + r_reg_kernel.y - reg_kernels.height() / 2,
                                       c) *
                            reg_kernels(r_reg_kernel.x, r_reg_kernel.y, n);
        Func rKTrKx0("rK^TKx0");
        rKTrKx0(x, y, c, n) = 0.f;
        rKTrKx0(x, y, c, n) += rKx0(x + r_reg_kernel.x - reg_kernels.width()  / 2,
                                    y + r_reg_kernel.y - reg_kernels.height() / 2,
                                    c,
                                    n) *
                               reg_kernels(r_reg_kernel.x, r_reg_kernel.y, n);

        ATb(x, y, c) += rKTb(x, y, c, r_reg_kernel.z) * reg_kernel_weights(r_reg_kernel.z);
        Func ATAx0("A^TAx0");
        ATAx0(x, y, c) = KTKx0(x, y, c);
        ATAx0(x, y, c) += rKTrKx0(x, y, c, r_reg_kernel.z) * reg_kernel_weights(r_reg_kernel.z);

        Func r0("r0");
        r0(x, y, c) = ATb(x, y, c) - ATAx0(x, y, c);
        Func p0("p0");
        p0(x, y, c) = r0(x, y, c);
        Func f_xrp("f_xrp");
        f_xrp(x, y, c, n) = 0.f;
        f_xrp(x, y, c, 0) = x0(x, y, c);
        f_xrp(x, y, c, 1) = r0(x, y, c);
        f_xrp(x, y, c, 2) = p0(x, y, c);

        xrp(x, y, c, n) = f_xrp(x, y, c, n);

        if (auto_schedule) {
            blurred.dim(0).set_bounds_estimate(0, 320);
            blurred.dim(1).set_bounds_estimate(0, 240);
            blurred.dim(2).set_bounds_estimate(0, 3);

            x0.dim(0).set_bounds_estimate(0, 320);
            x0.dim(1).set_bounds_estimate(0, 240);
            x0.dim(2).set_bounds_estimate(0, 3);

            kernel.dim(0).set_bounds_estimate(0, 5);
            kernel.dim(1).set_bounds_estimate(0, 5);

            reg_kernel_weights.dim(0).set_bounds_estimate(0, 2);

            reg_kernels.dim(0).set_bounds_estimate(0, 3);
            reg_kernels.dim(1).set_bounds_estimate(0, 3);
            reg_kernels.dim(2).set_bounds_estimate(0, 2);

            xrp.estimate(x, 0, 320)
               .estimate(y, 0, 240)
               .estimate(c, 0, 3)
               .estimate(n, 0, 3);
        } else {
            compute_all_root(xrp);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgInitGenerator, deconv_cg_init_forward)

#include "gradient_helpers.h"

#include "algorithms/deconv_cg_init.h"

namespace gradient_apps {

class DeconvCgInitForwardGenerator
  : public Generator<DeconvCgInitForwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  x0{"x0", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernel", 3};
    Output<Buffer<float>> xrp{"xrp", 4};

    void generate() {
        auto func_map = deconv_cg_init(blurred, x0, kernel, reg_kernel_weights, reg_kernels);
        assert(func_map.find("xrp") != func_map.end());
        xrp(x, y, c, n) = func_map["xrp"](x, y, c, n);

        if (auto_schedule) {
            blurred.dim(0).set_bounds_estimate(0, 320);
            blurred.dim(1).set_bounds_estimate(0, 240);
            blurred.dim(2).set_bounds_estimate(0, 3);

            x0.dim(0).set_bounds_estimate(0, 320);
            x0.dim(1).set_bounds_estimate(0, 240);
            x0.dim(2).set_bounds_estimate(0, 3);

            kernel.dim(0).set_bounds_estimate(0, 7);
            kernel.dim(1).set_bounds_estimate(0, 7);

            reg_kernel_weights.dim(0).set_bounds_estimate(0, 2);

            reg_kernels.dim(0).set_bounds_estimate(0, 3);
            reg_kernels.dim(1).set_bounds_estimate(0, 3);
            reg_kernels.dim(2).set_bounds_estimate(0, 2);

            xrp.estimate(x, 0, 320)
               .estimate(y, 0, 240)
               .estimate(c, 0, 3)
               .estimate(n, 0, 3);
        } else {
            int tile_width = 64, tile_height = 16;

            Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
            compute_all_root(xrp);
            Func Kx0 = func_map["Kx0"];
            Kx0.update()
               .tile(x, y, xo, yo, xi, yi, tile_width, tile_height)
               .parallel(yo)
               .vectorize(xi, 16);
            Func KTKx0 = func_map["KTKx0"];
            KTKx0.update()
                 .tile(x, y, xo, yo, xi, yi, tile_width, tile_height)
                 .parallel(yo)
                 .vectorize(xi, 16);
            Func rKx0 = func_map["rKx0"];
            rKx0.update()
                .tile(x, y, xo, yo, xi, yi, tile_width, tile_height)
                .parallel(yo)
                .vectorize(xi, 16);
            Func rKTrKx0 = func_map["rKTrKx0"];
            rKTrKx0.update()
                   .tile(x, y, xo, yo, xi, yi, tile_width, tile_height)
                   .parallel(yo)
                   .vectorize(xi, 16);
            Func KTb = func_map["KTb"];
            KTb.update()
               .tile(x, y, xo, yo, xi, yi, tile_width, tile_height)
               .parallel(yo)
               .vectorize(xi, 16);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgInitForwardGenerator, deconv_cg_init_forward)
#include "gradient_helpers.h"

#include "algorithms/deconv_cg_init.h"

namespace gradient_apps {

class DeconvCgInitBackwardGenerator
  : public Generator<DeconvCgInitBackwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  x0{"x0", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernel", 3};
    Input<Buffer<float>>  reg_target_kernels{"reg_target_kernels", 3};
    Input<Buffer<float>>  d_xrp{"d_xrp", 4};
    Output<Buffer<float>> d_reg_kernel_weights{"d_reg_kernel_weights", 1};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernels", 3};
    Output<Buffer<float>> d_reg_target_kernels{"d_reg_target_kernels", 3};

    void generate() {
        auto func_map = deconv_cg_init(blurred, x0, kernel, reg_kernel_weights, reg_kernels, reg_target_kernels);
        Func xrp = func_map["xrp"];
        Func reg_kernel_weights_func = func_map["reg_kernel_weights_func"];
        Func reg_kernels_func = func_map["reg_kernels_func"];
        Func reg_target_kernels_func = func_map["reg_target_kernels_func"];
        Derivative d = propagate_adjoints(
            xrp,
            d_xrp,
            {{d_xrp.dim(0).min(), d_xrp.dim(0).max()},
             {d_xrp.dim(1).min(), d_xrp.dim(1).max()},
             {d_xrp.dim(2).min(), d_xrp.dim(2).max()},
             {d_xrp.dim(3).min(), d_xrp.dim(3).max()}}
        );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        if (adjoints.find(FuncKey{reg_kernel_weights_func.name(), -1}) != adjoints.end()) {
            d_reg_kernel_weights(n) = adjoints[FuncKey{reg_kernel_weights_func.name(), -1}](n);
        } else {
            d_reg_kernel_weights(n) = 0.f;
        }
        if (adjoints.find(FuncKey{reg_kernels_func.name(), -1}) != adjoints.end()) {
            d_reg_kernels(x, y, n) = adjoints[FuncKey{reg_kernels_func.name(), -1}](x, y, n);
        } else {
            d_reg_kernels(x, y, n) = 0.f;
        }
        if (adjoints.find(FuncKey{reg_target_kernels_func.name(), -1}) != adjoints.end()) {
            d_reg_target_kernels(x, y, n) = adjoints[FuncKey{reg_kernels_func.name(), -1}](x, y, n);
        } else {
            assert(false);
            d_reg_target_kernels(x, y, n) = 0.f;
        }

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

            d_xrp.dim(0).set_bounds_estimate(0, 320);
            d_xrp.dim(1).set_bounds_estimate(0, 240);
            d_xrp.dim(2).set_bounds_estimate(0, 3);
            d_xrp.dim(3).set_bounds_estimate(0, 3);

            d_reg_kernel_weights.estimate(n, 0, 2);
            d_reg_kernels.estimate(x, 0, 3)
                         .estimate(y, 0, 3)
                         .estimate(n, 0, 2);
        } else {
            int tile_width = 64, tile_height = 16;

            compute_all_root(d_reg_kernel_weights);
            compute_all_root(d_reg_kernels);

            Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
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
            Func rKTb = func_map["rKTb"];
            rKTb.update()
                .tile(x, y, xo, yo, xi, yi, tile_width, tile_height)
                .parallel(yo)
                .vectorize(xi, 16);

            auto deps = get_deps({d_reg_kernel_weights, d_reg_kernels});
            Func d_rKx0_1 = Func(deps["rKx0_1_d_def__"]);
            d_rKx0_1.update()
                    .tile(x, y, xo, yo, xi, yi, tile_width, tile_height)
                    .parallel(yo)
                    .vectorize(xi, 16);

            Func d_rKw = Func(deps["reg_kernels_func_0_d_def__"]);
            auto d_rKw_r0 = d_rKw.rvars(0);
            auto d_rKw_r1 = d_rKw.rvars(1);

            Var xy("xy"), xyn("xyn");
            RVar rxo("rxo"), rxi("rxi");
            Var rxo_f("ryo_f"), rxi_f("ryi_f");
            d_rKw.compute_root()
                 .update(0)
                 .split(d_rKw_r0[0], rxo, rxi, tile_width);
            Func d_rKw0_ryo = d_rKw.update()
                                   .rfactor({{rxo, rxo_f}});
            d_rKw.update(1)
                 .split(d_rKw_r1[0], rxo, rxi, tile_width);
            Func d_rKw1_ryo = d_rKw.update(1)
                                   .rfactor({{rxo, rxo_f}});

            d_rKw.update(0)
                 .fuse(x, y, xy)
                 .fuse(xy, n, xyn)
                 .parallel(xyn);
            d_rKw.update(1)
                 .fuse(x, y, xy)
                 .fuse(xy, n, xyn)
                 .parallel(xyn);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgInitBackwardGenerator, deconv_cg_init_backward)

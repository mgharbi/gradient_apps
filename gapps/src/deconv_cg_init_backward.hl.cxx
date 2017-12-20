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
    Input<Buffer<float>>  precond_kernel{"precond_kernel", 2};
    Input<Buffer<float>>  w_kernel{"w_kernel", 3};
    Input<Buffer<float>>  w_reg_kernels{"w_reg_kernels", 4};
    Input<Buffer<float>>  d_xrp{"d_xrp", 4};
    Output<Buffer<float>> d_x0{"d_x0", 3};
    Output<Buffer<float>> d_reg_kernel_weights{"d_reg_kernel_weights", 1};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernels", 3};
    Output<Buffer<float>> d_reg_target_kernels{"d_reg_target_kernels", 3};
    Output<Buffer<float>> d_precond_kernel{"d_precond_kernel", 2};
    Output<Buffer<float>> d_w_kernel{"d_w_kernel", 3};
    Output<Buffer<float>> d_w_reg_kernels{"d_w_reg_kernels", 4};

    void generate() {
        auto func_map = deconv_cg_init(blurred, x0, kernel,
            reg_kernel_weights, reg_kernels, reg_target_kernels,
            precond_kernel, w_kernel, w_reg_kernels);
        Func xrp = func_map["xrp"];
        Func x0_func = func_map["x0_func"];
        Func reg_kernel_weights_func = func_map["reg_kernel_weights_func"];
        Func reg_kernels_func = func_map["reg_kernels_func"];
        Func reg_target_kernels_func = func_map["reg_target_kernels_func"];
        Func precond_kernel_func = func_map["precond_kernel_func"];
        Func w_kernel_func = func_map["w_kernel_func"];
        Func w_reg_kernels_func = func_map["w_reg_kernels_func"];
        Derivative d = propagate_adjoints(
            xrp,
            d_xrp,
            {{d_xrp.dim(0).min(), d_xrp.dim(0).max()},
             {d_xrp.dim(1).min(), d_xrp.dim(1).max()},
             {d_xrp.dim(2).min(), d_xrp.dim(2).max()},
             {d_xrp.dim(3).min(), d_xrp.dim(3).max()}}
        );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, x0_func, d_x0);
        assign_gradient(adjoints, reg_kernel_weights_func, d_reg_kernel_weights);
        assign_gradient(adjoints, reg_kernels_func, d_reg_kernels);
        assign_gradient(adjoints, reg_target_kernels_func, d_reg_target_kernels);
        assign_gradient(adjoints, precond_kernel_func, d_precond_kernel);
        assign_gradient(adjoints, w_kernel_func, d_w_kernel);
        assign_gradient(adjoints, w_reg_kernels_func, d_w_reg_kernels);

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
            auto func_map = get_deps({d_x0,
                                      d_reg_kernel_weights,
                                      d_reg_kernels,
                                      d_reg_target_kernels,
                                      d_precond_kernel,
                                      d_w_kernel,
                                      d_w_reg_kernels});
            compute_all_root(d_x0);
            compute_all_root(d_reg_kernel_weights);
            compute_all_root(d_reg_kernels);
            compute_all_root(d_reg_target_kernels);
            compute_all_root(d_precond_kernel);
            compute_all_root(d_w_kernel);
            compute_all_root(d_w_reg_kernels);

            Func Kx0 = Func(func_map["Kx0"]);
            Kx0.update()
               .parallel(y)
               .vectorize(x, 16);
            Func KTWKx0 = Func(func_map["K^TWKx0"]);
            KTWKx0.update()
                  .parallel(y)
                  .vectorize(x, 16);
            Func KTWb = Func(func_map["K^TWb"]);
            KTWb.update()
                .parallel(y)
                .vectorize(x, 16);
            assert(func_map.find("rKx0") != func_map.end());
            Func rKx0 = Func(func_map["rKx0"]);
            rKx0.update()
                .parallel(y)
                .vectorize(x, 16);
            assert(func_map.find("rK^TWrKx0") != func_map.end());
            Func rKTWrKx0 = Func(func_map["rK^TWrKx0"]);
            rKTWrKx0.update()
                    .parallel(y)
                    .vectorize(x, 16);
            assert(func_map.find("rK^TWb") != func_map.end());
            Func rKTWb = Func(func_map["rK^TWb"]);
            rKTWb.update()
                 .parallel(y)
                 .vectorize(x, 16);
            Func Pr0 = Func(func_map["Pr0"]);
            Pr0.update()
               .parallel(y)
               .vectorize(x, 16);
            
            Func d_WKx0 = Func(func_map["WKx0_0_d_def__"]);
            d_WKx0.update()
                  .parallel(y)
                  .vectorize(x, 16);
            Func d_WrKx0 = Func(func_map["WrKx0_0_d_def__"]);
            d_WrKx0.update()
                   .parallel(y)
                   .vectorize(x, 16);
            Func d_repeat_edge_1 = Func(func_map["repeat_edge$1_0_d_def__"]);
            d_repeat_edge_1.update(0)
                           .parallel(y)
                           .vectorize(x, 16);
            d_repeat_edge_1.update(1)
                           .parallel(y)
                           .vectorize(x, 16);

            Func d_rKw = Func(func_map["reg_kernels_func_0_d_def__"]);
            auto d_rKw_r0 = d_rKw.rvars(0);
            auto d_rKw_r1 = d_rKw.rvars(1);

            Var xy("xy"), xyn("xyn");
            RVar rxo("rxo"), rxi("rxi");
            Var rxi_f("rxi_f");
            d_rKw.compute_root()
                 .update(0)
                 .split(d_rKw_r0[0], rxo, rxi, 16);
            Func d_rKw0_rxi = d_rKw.update()
                                   .rfactor({{rxi, rxi_f}});
            d_rKw.update(1)
                 .split(d_rKw_r1[0], rxo, rxi, 16);
            Func d_rKw1_rxi = d_rKw.update(1)
                                   .rfactor({{rxi, rxi_f}});

            d_rKw.update(0)
                 .fuse(x, y, xy)
                 .fuse(xy, n, xyn)
                 .parallel(xyn);
            d_rKw.update(1)
                 .fuse(x, y, xy)
                 .fuse(xy, n, xyn)
                 .parallel(xyn);

            d_rKw0_rxi.compute_at(d_rKw, xyn)
                      .update()
                      .vectorize(rxi_f, 16);

            d_rKw1_rxi.compute_at(d_rKw, xyn)
                      .update()
                      .vectorize(rxi_f, 16);

            Func d_rtk = Func(func_map["reg_target_kernel_func_0_d_def__"]);
            auto d_rtk_r0 = d_rtk.rvars(0);

            d_rtk.compute_root()
                 .update(0)
                 .split(d_rtk_r0[0], rxo, rxi, 16);
            Func d_rtk_rxo = d_rtk.update()
                                  .rfactor({{rxi, rxi_f}});

            d_rtk.update(0)
                 .fuse(x, y, xy)
                 .fuse(xy, n, xyn)
                 .parallel(xyn);

            d_rtk_rxo.compute_at(d_rtk, xyn)
                     .update()
                     .vectorize(rxi_f, 16);

            Func d_wkb = Func(func_map["wkb_0_d_def__"]);
            d_wkb.update()
                 .parallel(y)
                 .vectorize(x, 16);

            Func d_wrkb = Func(func_map["wrkb_0_d_def__"]);
            d_wrkb.update()
                  .parallel(y)
                  .vectorize(x, 16);

            Func d_Pr0 = Func(func_map["Pr0_1_d_def__"]);
            d_Pr0.update()
                 .parallel(y)
                 .vectorize(x, 16);

            Func d_r = Func(func_map["repeat_edge$4_0_d_def__"]);
            d_r.update()
               .parallel(y)
               .vectorize(x, 16);

            Func d_pk = Func(func_map["precond_kernel_func_0_d_def__"]);
            auto d_pk_r0 = d_pk.rvars(0);
            auto d_pk_r1 = d_pk.rvars(1);
            d_pk.compute_root()
                .update(0)
                .split(d_pk_r0[0], rxo, rxi, 16);
            Func d_pk0_rxi = d_pk.update(0)
                                 .rfactor({{rxi, rxi_f}});
            d_pk.compute_root()
                .update(1)
                .split(d_pk_r1[0], rxo, rxi, 16);
            Func d_pk1_rxi = d_pk.update(1)
                                 .rfactor({{rxi, rxi_f}});

            d_pk.update(0)
                .fuse(x, y, xy)
                .parallel(xy);

            d_pk.update(1)
                .fuse(x, y, xy)
                .parallel(xy);

            d_pk0_rxi.compute_at(d_pk, xy)
                     .update()
                     .vectorize(rxi_f, 16);

            d_pk1_rxi.compute_at(d_pk, xy)
                     .update()
                     .vectorize(rxi_f, 16);

        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgInitBackwardGenerator, deconv_cg_init_backward)

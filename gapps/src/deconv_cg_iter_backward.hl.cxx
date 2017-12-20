#include "gradient_helpers.h"

#include "algorithms/deconv_cg_iter.h"

namespace gradient_apps {

class DeconvCgIterBackwardGenerator
  : public Generator<DeconvCgIterBackwardGenerator> {
public:
    Input<Buffer<float>>  xrp{"xrp", 4};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernel", 3};
    Input<Buffer<float>>  precond_kernel{"precond_kernel", 2};
    Input<Buffer<float>>  w_kernel{"w_kernel", 3};
    Input<Buffer<float>>  w_reg_kernels{"w_reg_kernels", 4};
    Input<Buffer<float>>  d_next_xrp{"d_next_xrp", 4};
    Output<Buffer<float>> d_xrp{"d_xrp", 4};
    Output<Buffer<float>> d_reg_kernel_weights{"d_reg_kernel_weights", 1};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernel", 3};
    Output<Buffer<float>> d_precond_kernel{"d_precond_kernel", 2};
    Output<Buffer<float>> d_w_kernel{"d_w_kernel", 3};
    Output<Buffer<float>> d_w_reg_kernels{"d_w_reg_kernels", 4};

    void generate() {
        auto func_map = deconv_cg_iter(xrp, kernel,
            reg_kernel_weights, reg_kernels, precond_kernel, w_kernel, w_reg_kernels);
        Func xrp_func = func_map["xrp_func"];
        Func reg_kernel_weights_func = func_map["reg_kernel_weights_func"];
        Func reg_kernels_func = func_map["reg_kernels_func"];
        Func precond_kernel_func = func_map["precond_kernel_func"];
        Func w_kernel_func = func_map["w_kernel_func"];
        Func w_reg_kernels_func = func_map["w_reg_kernels_func"];
        Func next_xrp = func_map["next_xrp"];
        Derivative d = propagate_adjoints(
            next_xrp,
            d_next_xrp,
            {{d_next_xrp.dim(0).min(), d_next_xrp.dim(0).max()},
             {d_next_xrp.dim(1).min(), d_next_xrp.dim(1).max()},
             {d_next_xrp.dim(2).min(), d_next_xrp.dim(2).max()},
             {d_next_xrp.dim(3).min(), d_next_xrp.dim(3).max()}}
        );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, xrp_func, d_xrp);
        assign_gradient(adjoints, reg_kernel_weights_func, d_reg_kernel_weights);
        assign_gradient(adjoints, reg_kernels_func, d_reg_kernels);
        assign_gradient(adjoints, precond_kernel_func, d_precond_kernel);
        assign_gradient(adjoints, w_kernel_func, d_w_kernel);
        assign_gradient(adjoints, w_reg_kernels_func, d_w_reg_kernels);

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

            d_next_xrp.dim(0).set_bounds_estimate(0, 320);
            d_next_xrp.dim(1).set_bounds_estimate(0, 240);
            d_next_xrp.dim(2).set_bounds_estimate(0, 3);
            d_next_xrp.dim(3).set_bounds_estimate(0, 3);

            d_xrp.estimate(x, 0, 320)
                 .estimate(y, 0, 240)
                 .estimate(c, 0, 3)
                 .estimate(n, 0, 3);

            d_reg_kernel_weights.estimate(n, 0, 2);
            d_reg_kernels.estimate(x, 0, 3)
                         .estimate(y, 0, 3)
                         .estimate(n, 0, 2);
        } else {
            auto func_map = get_deps({
                d_xrp,
                d_reg_kernel_weights,
                d_reg_kernels,
                d_precond_kernel,
                d_w_kernel,
                d_w_reg_kernels});
            compute_all_root(d_xrp);
            compute_all_root(d_reg_kernel_weights);
            compute_all_root(d_reg_kernels);
            compute_all_root(d_precond_kernel);
            compute_all_root(d_w_kernel);
            compute_all_root(d_w_reg_kernels);

            Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
            Func KTWKp = Func(func_map["K^TWKp"]);
            KTWKp.compute_root()
                 .parallel(y)
                 .vectorize(x, 16)
                 .update()
                 .parallel(y)
                 .vectorize(x, 16);
            Func Kp = Func(func_map["Kp"]);
            Kp.compute_root()
              .parallel(y)
              .vectorize(x, 16)
              .update()
              .parallel(y)
              .vectorize(x, 16);
            Func rKTWrKp = Func(func_map["rK^TWrKp"]);
            rKTWrKp.compute_root()
                   .parallel(y)
                   .vectorize(x, 16)
                   .update()
                   .parallel(y)
                   .vectorize(x, 16);
            Func rKp = Func(func_map["rKp"]);
            rKp.compute_root()
               .parallel(y)
               .vectorize(x, 16)
               .update()
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

            Func d_WKp = Func(func_map["WKp_0_d_def__"]);
            d_WKp.compute_root()
                 .parallel(y)
                 .vectorize(x, 16)
                 .update()
                 .parallel(y)
                 .vectorize(x, 16);

            Func d_WrKp = Func(func_map["WrKp_0_d_def__"]);
            d_WrKp.compute_root()
                  .parallel(y)
                  .vectorize(x, 16)
                  .update()
                  .parallel(y)
                  .vectorize(x, 16);

            Func d_p = Func(func_map["p_0_d_def__"]);
            d_p.compute_root()
               .parallel(y)
               .vectorize(x, 16)
               .update(0)
               .parallel(y)
               .vectorize(x, 16);
            d_p.update(1)
               .parallel(y)
               .vectorize(x, 16);
            d_p.update(2)
               .parallel(y)
               .vectorize(x, 16);
            d_p.update(3)
               .parallel(y)
               .vectorize(x, 16);
            d_p.update(4)
               .parallel(y)
               .vectorize(x, 16);

            Func d_rKw = Func(func_map["reg_kernels_func_0_d_def__"]);
            auto d_rKw_r0 = d_rKw.rvars(0);
            auto d_rKw_r1 = d_rKw.rvars(1);

            Var xy("xy"), xyn("xyn");
            RVar rxo("rxo"), rxi("rxi");
            Var rxi_f("ryi_f");
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

            Func d_Pr = Func(func_map["Pr_1_d_def__"]);
            d_Pr.update()
                .parallel(y)
                .vectorize(x, 16);

            Func d_r = Func(func_map["r_0_d_def__"]);
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

            // TODO: merge functions
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgIterBackwardGenerator, deconv_cg_iter_backward)


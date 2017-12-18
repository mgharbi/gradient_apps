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
    Input<Buffer<float>>  d_next_xrp{"d_next_xrp", 4};
    Output<Buffer<float>> d_xrp{"d_xrp", 4};
    Output<Buffer<float>> d_reg_kernel_weights{"d_reg_kernel_weights", 1};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernel", 3};

    void generate() {
        auto func_map = deconv_cg_iter(xrp, kernel, reg_kernel_weights, reg_kernels);
        Func xrp_func = func_map["xrp_func"];
        Func reg_kernel_weights_func = func_map["reg_kernel_weights_func"];
        Func reg_kernels_func = func_map["reg_kernels_func"];
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
        if (adjoints.find(FuncKey{xrp_func.name(), -1}) != adjoints.end()) {
            d_xrp(x, y, c, n) = adjoints[FuncKey{xrp_func.name(), -1}](x, y, c, n);
        } else {
            d_xrp(x, y, c, n) = 0.f;
        }
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
            compute_all_root(d_xrp);
            compute_all_root(d_reg_kernel_weights);
            compute_all_root(d_reg_kernels);

            int tile_width = 32, tile_height = 16;
            Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
            Func KTKp = func_map["KTKp"];
            KTKp.compute_root()
                .parallel(y)
                .vectorize(x, 16)
                .update()
                .parallel(y)
                .vectorize(x, 16);
            Func Kp = func_map["Kp"];
            Kp.compute_root()
              .parallel(y)
              .vectorize(x, 16)
              .update()
              .parallel(y)
              .vectorize(x, 16);
            Func rKTrKp = func_map["rKTrKp"];
            rKTrKp.compute_root()
                  .parallel(y)
                  .vectorize(x, 16)
                  .update()
                  .parallel(y)
                  .vectorize(x, 16);
            Func rKp = func_map["rKp"];
            rKp.compute_root()
               .parallel(y)
               .vectorize(x, 16)
               .update()
               .parallel(y)
               .vectorize(x, 16);

            auto deps = get_deps({d_xrp, d_reg_kernel_weights, d_reg_kernels});
            Func d_Kp_1 = Func(deps["Kp_1_d_def__"]);
            d_Kp_1.compute_root()
                  .parallel(y)
                  .vectorize(x, 16)
                  .update()
                  .parallel(y)
                  .vectorize(x, 16);
            Func d_rKp_1 = Func(deps["rKp_1_d_def__"]);
            d_rKp_1.compute_root()
                   .parallel(y)
                   .vectorize(x, 16)
                   .update()
                   .parallel(y)
                   .vectorize(x, 16);
            Func d_p0 = Func(deps["p_0_d_def__"]);
            d_p0.compute_root()
                .parallel(y)
                .vectorize(x, 16)
                .update(0)
                .parallel(y)
                .vectorize(x, 16);
            d_p0.update(1)
                .parallel(y)
                .vectorize(x, 16);
            d_p0.update(2)
                .parallel(y)
                .vectorize(x, 16);
            d_p0.update(3)
                .parallel(y)
                .vectorize(x, 16);
            d_p0.update(4)
                .parallel(y)
                .vectorize(x, 16);

            // wtf is this??
            Func repeat_edge22 = Func(deps["repeat_edge$22"]);
            repeat_edge22.compute_root()
                         .parallel(y)
                         .vectorize(x, 16);

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

            // TODO: merge more functions
            //Func xrp_func = deps["xrp_func"];
            Func(deps["xrp_func"]).compute_inline();
            Func(deps["kernel_im"]).compute_inline();
            Func(deps["xrp_im"]).compute_inline();
            Func(deps["repeat_edge"]).compute_inline();
            Func(deps["p"]).compute_inline();
            Func(deps["reg_kernel_im"]).compute_inline();
            Func(deps["reg_kernels_func"]).compute_inline();
            Func(deps["reg_kernel_weights_im"]).compute_inline();
            Func(deps["reg_kernel_weights_func"]).compute_inline();
            Func(deps["r"]).compute_inline();
            //Func(deps["A^TAp"]).compute_inline();
            Func(deps["d_next_xrp_im"]).compute_inline();
            Func(deps["next_xrp_3_d_def__"]).compute_inline();
            Func(deps["repeat_edge$1"]).compute_inline();
            Func(deps["next_xrp_3_d_def___ce"]).compute_inline();
            Func(deps["next_p_0_d_def__"]).compute_inline();
            Func(deps["repeat_edge$4"]).compute_inline();
            Func(deps["next_p_0_d_def___ce"]).compute_inline();
            Func(deps["next_r"]).compute_inline();
            Func(deps["next_xrp_2_d_def__$1"]).compute_inline();
            //Func(deps["next_r_0_d_def__"]).compute_inline();
            //Func(deps["repeat_edge$7"]).compute_inline();
            //Func(deps["next_r_0_d_def___ce"]).compute_inline();
            //Func(deps["next_xrp_1_d_def__$1"]).compute_inline();
            //Func(deps["next_x_0_d_def__"]).compute_inline();
            //Func(deps["repeat_edge$2"]).compute_inline(); 
            //Func(deps["next_x_0_d_def___ce"]).compute_inline();
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgIterBackwardGenerator, deconv_cg_iter_backward)


#include "gradient_helpers.h"

#include "algorithms/deconv_cg_init.h"

namespace gradient_apps {

class DeconvCgInitBackwardGenerator
  : public Generator<DeconvCgInitBackwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  x0{"x0", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  data_kernel_weights{"data_kernel_weights", 1};
    Input<Buffer<float>>  data_kernels{"data_kernels", 3};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  reg_targets{"reg_targets", 4};
    Input<Buffer<float>>  w_data{"w_data", 4};
    Input<Buffer<float>>  w_reg{"w_reg", 4};
    Input<Buffer<float>>  d_xrp{"d_xrp", 4};
    Output<Buffer<float>> d_x0{"d_x0", 3};
    Output<Buffer<float>> d_data_kernel_weights{"d_data_kernel_weights", 1};
    Output<Buffer<float>> d_data_kernels{"d_data_kernels", 3};
    Output<Buffer<float>> d_reg_kernel_weights{"d_reg_kernel_weights", 1};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernels", 3};
    Output<Buffer<float>> d_reg_targets{"d_reg_targets", 4};
    Output<Buffer<float>> d_w_data{"d_w_data", 4};
    Output<Buffer<float>> d_w_reg{"d_w_reg", 4};

    void generate() {
        auto func_map = deconv_cg_init(blurred, x0, kernel,
            data_kernel_weights, data_kernels,
            reg_kernel_weights, reg_kernels, reg_targets,
            w_data, w_reg);
        Func xrp = func_map["xrp"];
        Func x0_func = func_map["x0_func"];
        Func data_kernel_weights_func = func_map["data_kernel_weights_func"];
        Func data_kernels_func = func_map["data_kernels_func"];
        Func reg_kernel_weights_func = func_map["reg_kernel_weights_func"];
        Func reg_kernels_func = func_map["reg_kernels_func"];
        Func reg_targets_func = func_map["reg_targets_func"];
        Func w_data_func = func_map["w_data_func"];
        Func w_reg_func = func_map["w_reg_func"];
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
        assign_gradient(adjoints, data_kernel_weights_func, d_data_kernel_weights);
        assign_gradient(adjoints, data_kernels_func, d_data_kernels);
        assign_gradient(adjoints, reg_kernel_weights_func, d_reg_kernel_weights);
        assign_gradient(adjoints, reg_kernels_func, d_reg_kernels);
        assign_gradient(adjoints, reg_targets_func, d_reg_targets);
        assign_gradient(adjoints, w_data_func, d_w_data);
        assign_gradient(adjoints, w_reg_func, d_w_reg);

        if (auto_schedule) {} else {
            std::vector<Func> funcs{d_x0,
                                    d_data_kernel_weights,
                                    d_data_kernels,
                                    d_reg_kernel_weights,
                                    d_reg_kernels,
                                    d_reg_targets,
                                    d_w_data,
                                    d_w_reg};
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            simple_autoschedule(funcs,
                                {{"blurred.min.0", 0},
                                 {"blurred.min.1", 0},
                                 {"blurred.min.2", 0},
                                 {"blurred.extent.0", 256},
                                 {"blurred.extent.1", 256},
                                 {"blurred.extent.2", 3},
                                 {"x0.min.0", 0},
                                 {"x0.min.1", 0},
                                 {"x0.min.2", 0},
                                 {"x0.extent.0", 256},
                                 {"x0.extent.1", 256},
                                 {"x0.extent.2", 3},
                                 {"kernel.min.0", 0},
                                 {"kernel.min.1", 0},
                                 {"kernel.extent.0", 11},
                                 {"kernel.extent.1", 11},
                                 {"data_kernel_weights.min.0", 0},
                                 {"data_kernel_weights.extent.0", 5},
                                 {"data_kernels.min.0", 0},
                                 {"data_kernels.min.1", 0},
                                 {"data_kernels.min.2", 0},
                                 {"data_kernels.extent.0", 5},
                                 {"data_kernels.extent.1", 5},
                                 {"data_kernels.extent.2", 5},
                                 {"reg_kernel_weights.min.0", 0},
                                 {"reg_kernel_weights.extent.0", 5},
                                 {"reg_kernels.min.0", 0},
                                 {"reg_kernels.min.1", 0},
                                 {"reg_kernels.min.2", 0},
                                 {"reg_kernels.extent.0", 5},
                                 {"reg_kernels.extent.1", 5},
                                 {"reg_kernels.extent.2", 5},
                                 {"reg_targets.min.0", 0},
                                 {"reg_targets.min.1", 0},
                                 {"reg_targets.min.2", 0},
                                 {"reg_targets.min.3", 0},
                                 {"reg_targets.extent.0", 256},
                                 {"reg_targets.extent.1", 256},
                                 {"reg_targets.extent.2", 3},
                                 {"reg_targets.extent.3", 5},
                                 {"w_data.min.0", 0},
                                 {"w_data.min.1", 0},
                                 {"w_data.min.2", 0},
                                 {"w_data.min.3", 0},
                                 {"w_data.extent.0", 256},
                                 {"w_data.extent.1", 256},
                                 {"w_data.extent.2", 3},
                                 {"w_data.extent.3", 5},
                                 {"w_reg_kernels.min.0", 0},
                                 {"w_reg_kernels.min.1", 0},
                                 {"w_reg_kernels.min.2", 0},
                                 {"w_reg_kernels.min.3", 0},
                                 {"w_reg_kernels.extent.0", 256},
                                 {"w_reg_kernels.extent.1", 256},
                                 {"w_reg_kernels.extent.2", 3},
                                 {"w_reg_kernels.extent.3", 5},
                                 {"d_xrp.min.0", 0},
                                 {"d_xrp.min.1", 0},
                                 {"d_xrp.min.2", 0},
                                 {"d_xrp.min.3", 0},
                                 {"d_xrp.extent.0", 256},
                                 {"d_xrp.extent.1", 256},
                                 {"d_xrp.extent.2", 3},
                                 {"d_xrp.extent.3", 3}
                                },
                                {{{0, 255}, // x0
                                  {0, 255},
                                  {0, 2}},
                                 {{0, 4}}, // data_kernel_weights
                                 {{0, 4},  // data_kernels
                                  {0, 4},
                                  {0, 4}},
                                 {{0, 4}}, // reg_kernel_weights
                                 {{0, 4},  // reg_kernels
                                  {0, 4},
                                  {0, 4}},
                                 {{0, 255}, // reg_targets
                                  {0, 255},
                                  {0, 2},
                                  {0, 4}},
                                 {{0, 255}, // w_data
                                  {0, 255},
                                  {0, 2},
                                  {0, 4}},
                                 {{0, 255}, // w_reg
                                  {0, 255},
                                  {0, 2},
                                  {0, 4}}},
                                  options);

#if 0
            auto func_map = get_deps({d_x0,
                                      d_reg_kernel_weights,
                                      d_reg_kernels,
                                      d_precond_kernel,
                                      d_w_kernel,
                                      d_w_reg_kernels});
            print_func(Func(func_map["reg_kernels_func_0_d_def__"]), false, false, true, 1);
            compute_all_root(d_x0);
            compute_all_root(d_reg_kernel_weights);
            compute_all_root(d_reg_kernels);
            compute_all_root(d_precond_kernel);
            compute_all_root(d_w_kernel);
            compute_all_root(d_w_reg_kernels);

            Func Kx0 = get_func(func_map, "Kx0");
            Kx0.update()
               .parallel(y)
               .vectorize(x, 16);
            Func KTWKx0 = get_func(func_map, "K^TWKx0");
            KTWKx0.update()
                  .parallel(y)
                  .vectorize(x, 16);
            Func rKx0 = get_func(func_map, "rKx0");
            rKx0.update()
                .parallel(y)
                .vectorize(x, 16);
            Func rKTWrKx0 = get_func(func_map, "rK^TWrKx0");
            rKTWrKx0.update()
                    .parallel(y)
                    .vectorize(x, 16);
            Func Pr0 = get_func(func_map, "Pr0");
            Pr0.update()
               .parallel(y)
               .vectorize(x, 16);

            Func d_WKx0 = get_func(func_map, "WKx0_0_d_def__");
            d_WKx0.update()
                  .parallel(y)
                  .vectorize(x, 16);
            Func d_WrKx0 = get_func(func_map, "WrKx0_0_d_def__");
            d_WrKx0.update()
                   .parallel(y)
                   .vectorize(x, 16);
            Func d_repeat_edge_1 = get_func(func_map, "repeat_edge$1_0_d_def__");
            d_repeat_edge_1.update(0)
                           .parallel(y)
                           .vectorize(x, 16);
            d_repeat_edge_1.update(1)
                           .parallel(y)
                           .vectorize(x, 16);

            Func d_rKw = get_func(func_map, "reg_kernels_func_0_d_def__");
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

            Func d_wkb = get_func(func_map, "wkb_0_d_def__");
            d_wkb.update()
                 .parallel(y)
                 .vectorize(x, 16);

            Func d_Pr0 = get_func(func_map, "Pr0_1_d_def__");
            d_Pr0.update()
                 .parallel(y)
                 .vectorize(x, 16);

            Func d_r = get_func(func_map, "repeat_edge$4_0_d_def__");
            d_r.update()
               .parallel(y)
               .vectorize(x, 16);

            Func d_pk = get_func(func_map, "precond_kernel_func_0_d_def__");
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
#endif
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgInitBackwardGenerator, deconv_cg_init_backward)

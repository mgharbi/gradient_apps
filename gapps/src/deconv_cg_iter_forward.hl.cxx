#include "gradient_helpers.h"

#include "algorithms/deconv_cg_iter.h"

namespace gradient_apps {

class DeconvCgIterForwardGenerator
  : public Generator<DeconvCgIterForwardGenerator> {
public:
    Input<Buffer<float>>  xrp{"xrp", 4};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  data_kernel_weights{"data_kernel_weights", 1};
    Input<Buffer<float>>  data_kernels{"data_kernels", 3};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  w_data{"w_data", 4};
    Input<Buffer<float>>  w_reg{"w_reg", 4};
    Output<Buffer<float>> next_xrp{"next_xrp", 4};

    void generate() {
        auto func_map = deconv_cg_iter(xrp, kernel,
            data_kernel_weights, data_kernels,
            reg_kernel_weights, reg_kernels,
            w_data, w_reg);
        next_xrp(x, y, c, n) = func_map["next_xrp"](x, y, c, n);

        if (auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            Func output = next_xrp;
            simple_autoschedule(output,
                                {{"xrp.min.0", 0},
                                 {"xrp.min.1", 0},
                                 {"xrp.min.2", 0},
                                 {"xrp.min.3", 0},
                                 {"xrp.extent.0", 256},
                                 {"xrp.extent.1", 256},
                                 {"xrp.extent.2", 3},
                                 {"xrp.extent.3", 3},
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
                                 {"w_reg.min.0", 0},
                                 {"w_reg.min.1", 0},
                                 {"w_reg.min.2", 0},
                                 {"w_reg.min.3", 0},
                                 {"w_reg.extent.0", 256},
                                 {"w_reg.extent.1", 256},
                                 {"w_reg.extent.2", 3},
                                 {"w_reg.extent.3", 5}
                                },
                                {{0, 255},
                                 {0, 255},
                                 {0, 2},
                                 {0, 3}},
                                options,
                                {"next_xrp$1"});
#if 0
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
#endif
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgIterForwardGenerator, deconv_cg_iter_forward)

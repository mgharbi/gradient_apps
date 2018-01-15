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
        Func xrp = deconv_cg_init(blurred, x0, kernel,
            data_kernel_weights, data_kernels,
            reg_kernel_weights, reg_kernels, reg_targets,
            w_data, w_reg);
        Derivative d = propagate_adjoints(
            xrp,
            d_xrp,
            {{d_xrp.dim(0).min(), d_xrp.dim(0).max()},
             {d_xrp.dim(1).min(), d_xrp.dim(1).max()},
             {d_xrp.dim(2).min(), d_xrp.dim(2).max()},
             {d_xrp.dim(3).min(), d_xrp.dim(3).max()}}
        );
        assign_gradient(d, x0, d_x0);
        assign_gradient(d, data_kernel_weights, d_data_kernel_weights);
        assign_gradient(d, data_kernels, d_data_kernels);
        assign_gradient(d, reg_kernel_weights, d_reg_kernel_weights);
        assign_gradient(d, reg_kernels, d_reg_kernels);
        assign_gradient(d, reg_targets, d_reg_targets);
        assign_gradient(d, w_data, d_w_data);
        assign_gradient(d, w_reg, d_w_reg);

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
                                 {"w_reg.min.0", 0},
                                 {"w_reg.min.1", 0},
                                 {"w_reg.min.2", 0},
                                 {"w_reg.min.3", 0},
                                 {"w_reg.extent.0", 256},
                                 {"w_reg.extent.1", 256},
                                 {"w_reg.extent.2", 3},
                                 {"w_reg.extent.3", 5},
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
                                  options,
                                 {"xrp_1_d_def__$1"});
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgInitBackwardGenerator, deconv_cg_init_backward)

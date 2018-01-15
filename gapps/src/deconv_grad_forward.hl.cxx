#include "gradient_helpers.h"

#include "algorithms/deconv_cg.h"

namespace gradient_apps {

class DeconvGradForwardGenerator
  : public Generator<DeconvGradForwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  xk{"xk", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  data_kernel_weights{"data_kernel_weights", 1};
    Input<Buffer<float>>  data_kernels{"data_kernels", 3};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  reg_powers{"reg_powers", 1};
    Input<Buffer<float>>  reg_targets{"reg_targets", 4};
    Output<Buffer<float>> output{"output", 3};

    void generate() {
        Func cost = deconv_cost(
            xk, blurred, kernel,
            data_kernel_weights, data_kernels,
            reg_kernel_weights, reg_kernels, reg_powers, reg_targets);
        Derivative d = propagate_adjoints(cost);
        Func grad = d(xk);
        output(x, y, c) = grad(x, y, c);

        if (auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            Func output_func = output;
            simple_autoschedule(output_func,
                                {
                                 {"blurred.min.0", 0},
                                 {"blurred.min.1", 0},
                                 {"blurred.min.2", 0},
                                 {"blurred.extent.0", 256},
                                 {"blurred.extent.1", 256},
                                 {"blurred.extent.2", 3},
                                 {"xk.min.0", 0},
                                 {"xk.min.1", 0},
                                 {"xk.min.2", 0},
                                 {"xk.extent.0", 256},
                                 {"xk.extent.1", 256},
                                 {"xk.extent.2", 3},
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
                                 {"reg_powers.min.0", 0},
                                 {"reg_powers.extent.0", 5},
                                 {"reg_targets.min.0", 0},
                                 {"reg_targets.min.1", 0},
                                 {"reg_targets.min.2", 0},
                                 {"reg_targets.min.3", 0},
                                 {"reg_targets.extent.0", 256},
                                 {"reg_targets.extent.1", 256},
                                 {"reg_targets.extent.2", 3},
                                 {"reg_targets.extent.3", 5},
                                },
                                {
                                 {{0, 255},
                                  {0, 255},
                                  {0, 2}},
                                },
                                options);
        }
    }
};

}  // end namespace gradient_apps


HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvGradForwardGenerator, deconv_grad_forward)

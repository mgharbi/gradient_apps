#include "gradient_helpers.h"

#include "algorithms/deconv_cg_weight.h"

namespace gradient_apps {

class DeconvCgWeightForwardGenerator
  : public Generator<DeconvCgWeightForwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  current{"current", 3};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  reg_targets{"reg_target_kernels", 4};
    Input<Buffer<float>>  reg_powers{"reg_powers", 1};
    Output<Buffer<float>> weights{"weights", 4};

    void generate() {
        auto func_map = deconv_cg_weight(blurred, current,
            reg_kernels, reg_targets, reg_powers);
        weights(x, y, c, n) = func_map["weights"](x, y, c, n);

        if (auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            Func weights_func = weights;
            simple_autoschedule(weights_func,
                                {{"blurred.min.0", 0},
                                 {"blurred.min.1", 0},
                                 {"blurred.min.2", 0},
                                 {"blurred.extent.0", 256},
                                 {"blurred.extent.1", 256},
                                 {"blurred.extent.2", 3},
                                 {"current.min.0", 0},
                                 {"current.min.1", 0},
                                 {"current.min.2", 0},
                                 {"current.extent.0", 256},
                                 {"current.extent.1", 256},
                                 {"current.extent.2", 3},
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
                                 {"reg_powers.min.0", 0},
                                 {"reg_powers.extent.0", 5}},
                               {{{0, 255},
                                 {0, 255},
                                 {0, 2},
                                 {0, 4}}},
                               options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgWeightForwardGenerator, deconv_cg_weight_forward)

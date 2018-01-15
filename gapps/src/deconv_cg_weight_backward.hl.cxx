#include "gradient_helpers.h"

#include "algorithms/deconv_cg_weight.h"

namespace gradient_apps {

class DeconvCgWeightBackwardGenerator
  : public Generator<DeconvCgWeightBackwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  current{"current", 3};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  reg_targets{"reg_targets", 4};
    Input<Buffer<float>>  reg_powers{"reg_powers", 1};
    Input<Buffer<float>>  d_weights{"d_weights", 4};
    Output<Buffer<float>> d_current{"d_current", 3};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernels", 3};
    Output<Buffer<float>> d_reg_targets{"d_reg_targets", 4};
    Output<Buffer<float>> d_reg_powers{"d_reg_powers", 1};

    void generate() {
        Func weights = deconv_cg_weight(blurred, current,
            reg_kernels, reg_targets, reg_powers);
        Derivative d = propagate_adjoints(
            weights,
            d_weights,
            {{d_weights.dim(0).min(), d_weights.dim(0).max()},
             {d_weights.dim(1).min(), d_weights.dim(1).max()},
             {d_weights.dim(2).min(), d_weights.dim(2).max()},
             {d_weights.dim(3).min(), d_weights.dim(3).max()}}
        );
        assign_gradient(d, current, d_current);
        assign_gradient(d, reg_kernels, d_reg_kernels);
        assign_gradient(d, reg_targets, d_reg_targets);
        assign_gradient(d, reg_powers, d_reg_powers);

        if (auto_schedule) {
        } else {
            std::vector<Func> funcs{d_current, d_reg_kernels, d_reg_targets, d_reg_powers};
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            simple_autoschedule(funcs,
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
                                 {"reg_powers.extent.0", 5},
                                 {"d_weights.min.0", 0},
                                 {"d_weights.min.1", 0},
                                 {"d_weights.min.2", 0},
                                 {"d_weights.min.3", 0},
                                 {"d_weights.extent.0", 256},
                                 {"d_weights.extent.1", 256},
                                 {"d_weights.extent.2", 3},
                                 {"d_weights.extent.3", 5}
                                },
                               {{{0, 255}, // current
                                 {0, 255},
                                 {0, 2}},
                                {{0, 4}, // reg_kernels
                                 {0, 4},
                                 {0, 4}},
                                {{0, 255}, // reg_targets
                                 {0, 255},
                                 {0, 2},
                                 {0, 5}},
                                {{0, 4}}  // reg_powers
                                },
                               options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgWeightBackwardGenerator, deconv_cg_weight_backward)

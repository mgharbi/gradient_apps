#include "gradient_helpers.h"

#include "algorithms/deconv_prior.h"

namespace gradient_apps {

class DeconvPriorBackwardGenerator
  : public Generator<DeconvPriorBackwardGenerator> {
public:
    Input<Buffer<float>>  f{"f", 3};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  thresholds{"thresholds", 1};
    Input<Buffer<float>>  d_weights{"d_weights", 4};
    Output<Buffer<float>> d_f{"d_f", 3};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernels", 3};
    Output<Buffer<float>> d_thresholds{"d_thresholds", 1};

    void generate() {
        Func weights = deconv_prior(f, reg_kernels, thresholds);
        Derivative d = propagate_adjoints(
            weights,
            d_weights,
            {{d_weights.dim(0).min(), d_weights.dim(0).max()},
             {d_weights.dim(1).min(), d_weights.dim(1).max()},
             {d_weights.dim(2).min(), d_weights.dim(2).max()},
             {d_weights.dim(3).min(), d_weights.dim(3).max()}}
        );
        assign_gradient(d, f, d_f);
        assign_gradient(d, reg_kernels, d_reg_kernels);
        assign_gradient(d, thresholds, d_thresholds);

        if (auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            options.gpu_tile_channel = 3;
            std::vector<Func> funcs{d_f, d_reg_kernels, d_thresholds};
            simple_autoschedule(funcs,
                                {{"f.min.0", 0},
                                 {"f.min.1", 0},
                                 {"f.min.2", 0},
                                 {"f.extent.0", 256},
                                 {"f.extent.1", 256},
                                 {"f.extent.2", 3},
                                 {"reg_kernels.min.0", 0},
                                 {"reg_kernels.min.1", 0},
                                 {"reg_kernels.min.2", 0},
                                 {"reg_kernels.extent.0", 5},
                                 {"reg_kernels.extent.1", 5},
                                 {"reg_kernels.extent.2", 5},
                                 {"thresholds.min.0", 0},
                                 {"thresholds.extent.0", 5},
                                 {"d_weights.min.0", 0},
                                 {"d_weights.min.1", 0},
                                 {"d_weights.min.2", 0},
                                 {"d_weights.min.3", 0},
                                 {"d_weights.extent.0", 256},
                                 {"d_weights.extent.1", 256},
                                 {"d_weights.extent.2", 3},
                                 {"d_weights.extent.3", 5}
                                },
                               {{{0, 255},
                                 {0, 255},
                                 {0, 2}},
                                {{0, 4},
                                 {0, 4},
                                 {0, 4}},
                                {{0, 4}}},
                               options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvPriorBackwardGenerator, deconv_prior_backward)

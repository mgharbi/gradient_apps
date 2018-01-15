#include "gradient_helpers.h"

#include "algorithms/deconv_prior.h"

namespace gradient_apps {

class DeconvPriorForwardGenerator
  : public Generator<DeconvPriorForwardGenerator> {
public:
    Input<Buffer<float>>  f{"f", 3};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  thresholds{"thresholds", 1};
    Output<Buffer<float>> weights{"weights", 4};

    void generate() {
        Func output = deconv_prior(f, reg_kernels, thresholds);
        weights(x, y, c, n) = output(x, y, c, n);

        if (auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            Func weights_func = weights;
            simple_autoschedule(weights_func,
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
                                 {"thresholds.extent.0", 5}},
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
    gradient_apps::DeconvPriorForwardGenerator, deconv_prior_forward)

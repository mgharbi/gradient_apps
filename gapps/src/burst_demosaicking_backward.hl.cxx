#include "algorithms/burst_demosaicking.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class BurstDemosaickingBackwardGenerator
  : public Generator<BurstDemosaickingBackwardGenerator> {
public:
    Input<Buffer<float>> inputs{"inputs", 3};
    Input<Buffer<float>> homographies{"homographies", 2};
    Input<Buffer<float>> reconstructed{"reconstructed", 3};
    Input<Buffer<float>>  gradient_weight{"gradient_weight", 1};
    Input<Buffer<float>> d_loss{"d_loss", 1};

    Output<Buffer<float>> d_homographies{"d_homographies", 2};
    Output<Buffer<float>> d_reconstructed{"d_reconstructed", 3};

    void generate() {
        std::map<std::string, Func> func_map = burst_demosaicking(
            inputs, homographies, reconstructed, gradient_weight);

        Func f_loss = func_map["loss"];
        Func f_homographies = func_map["homographies"];
        Func f_reconstructed = func_map["reconstructed"];

        Derivative d = propagate_adjoints(
            f_loss, d_loss,
            {{d_loss.dim(0).min(), d_loss.dim(0).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, f_homographies, d_homographies);
        assign_gradient(adjoints, f_reconstructed, d_reconstructed);

        SimpleAutoscheduleOptions options;
        options.gpu = get_target().has_gpu_feature();

        std::set<std::string> dont_inline = {};

        std::vector<Func> funcs{d_homographies, d_reconstructed};

        simple_autoschedule(funcs,
            {
            {"inputs.min.0", 0},
            {"inputs.min.1", 0},
            {"inputs.min.2", 0},
            {"inputs.extent.0", 256},
            {"inputs.extent.1", 256},
            {"inputs.extent.2", 5},
            {"homographies.min.0", 0},
            {"homographies.min.1", 0},
            {"homographies.extent.0", 8},
            {"homographies.extent.1", 5},
            {"reconstructed.min.0", 0},
            {"reconstructed.min.1", 0},
            {"reconstructed.min.2", 0},
            {"reconstructed.extent.0", 256},
            {"reconstructed.extent.1", 256},
            {"reconstructed.extent.2", 3},
            {"gradient_weight.min.0", 0},
            {"gradient_weight.extent.0", 0},
            {"d_loss.min.0", 0},
            {"d_loss.extent.0", 1},
            },
            {
              {{0, 7}, {0, 4}},
              {{0, 255}, {0, 255}, {0, 2}},
            },
            options,
            dont_inline);

    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BurstDemosaickingBackwardGenerator, 
    burst_demosaicking_backward)

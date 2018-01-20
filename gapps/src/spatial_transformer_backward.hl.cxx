#include "algorithms/spatial_transformer.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class SpatialTransformerBackwardGenerator
  : public Generator<SpatialTransformerBackwardGenerator> {
public:
    Input<Buffer<float>>   input{"input", 4};
    Input<Buffer<float>>   affine_mtx{"affine_mtx", 3};
    Input<Buffer<float>>   d_output{"d_output", 4};
    Output<Buffer<float>>  d_input{"d_input", 4};
    Output<Buffer<float>>  d_affine_mtx{"d_affine_mtx", 3};

    void generate() {
        Func output = spatial_transformer(input, affine_mtx);

        Derivative d = propagate_adjoints(
            output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}
             });
        assign_gradient(d, input, d_input);
        assign_gradient(d, affine_mtx, d_affine_mtx);

        SimpleAutoscheduleOptions options;
        options.gpu = get_target().has_gpu_feature();

        std::set<std::string> dont_inline = {};

        std::vector<Func> funcs{d_input, d_affine_mtx};

        simple_autoschedule(funcs,
            {
              {"input.min.0", 0},
              {"input.min.1", 0},
              {"input.min.2", 0},
              {"input.min.3", 0},
              {"input.extent.0", 256},
              {"input.extent.1", 256},
              {"input.extent.2", 3},
              {"input.extent.3", 16},
              {"affine_mtx.min.0", 0},
              {"affine_mtx.min.1", 0},
              {"affine_mtx.min.2", 0},
              {"affine_mtx.extent.0", 3},
              {"affine_mtx.extent.1", 2},
              {"affine_mtx.extent.2", 16},
              {"d_output.min.0", 0},
              {"d_output.min.1", 0},
              {"d_output.min.2", 0},
              {"d_output.min.3", 0},
              {"d_output.extent.0", 256},
              {"d_output.extent.1", 256},
              {"d_output.extent.2", 3},
              {"d_output.extent.3", 16}
            },
            {
              {{0, 255}, {0, 255}, {0, 2}, {0, 15}},
              {{0, 2}, {0, 1}, {0, 15}},
            },
            options,
            dont_inline);
        print_func(d(input));
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::SpatialTransformerBackwardGenerator, 
    spatial_transformer_backward)

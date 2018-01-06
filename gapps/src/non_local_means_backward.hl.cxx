#include "algorithms/non_local_means.h"
#include "gradient_helpers.h"

namespace gradient_apps {

class NonLocalMeansBackwardGenerator : public Generator<NonLocalMeansBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 3};       // x, y, channel
    Input<Buffer<float>>  feature_filter{"feature_filter", 4};
    Input<Buffer<float>>  patch_filter{"patch_filter", 2};
    Input<float>          inv_sigma{"inv_sigma"};
    Input<int>            search_area{"search_area"};
    Input<Buffer<float>>  d_output{"d_output", 3};

    Output<Buffer<float>> d_input{"d_input", 3};
    Output<Buffer<float>> d_feature_filter{"d_feature_filter", 4};
    Output<Buffer<float>> d_patch_filter{"d_patch_filter", 2};
    Output<Buffer<float>> d_inv_sigma{"d_inv_sigma", 0};

    void generate() {
        std::map<std::string, Func> func_map = non_local_means(
            input, feature_filter, patch_filter, inv_sigma, search_area);
        Func output = func_map["output"];
        Func clamped = func_map["clamped"];
        Func feature_filter_func = func_map["feature_filter_func"];
        Func patch_filter_func = func_map["patch_filter_func"];
        Func inv_sigma_func = func_map["inv_sigma_func"];
        Derivative d = propagate_adjoints(
            output,
            d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()}}
        );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, clamped, d_input);
        assign_gradient(adjoints, feature_filter_func, d_feature_filter);
        assign_gradient(adjoints, patch_filter_func, d_patch_filter);
        assign_gradient(adjoints, inv_sigma_func, d_inv_sigma);

        if(auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            std::vector<Func> funcs{d_input,
                                    d_feature_filter,
                                    d_patch_filter,
                                    d_inv_sigma};
            simple_autoschedule(funcs,
                                {{"input.min.0", 0},
                                 {"input.min.1", 0},
                                 {"input.min.2", 0},
                                 {"input.extent.0", 512},
                                 {"input.extent.1", 512},
                                 {"input.extent.2", 3},
                                 {"feature_filter.min.0", 0},
                                 {"feature_filter.min.1", 0},
                                 {"feature_filter.min.2", 0},
                                 {"feature_filter.min.3", 0},
                                 {"feature_filter.extent.0", 5},
                                 {"feature_filter.extent.1", 5},
                                 {"feature_filter.extent.2", 3},
                                 {"feature_filter.extent.3", 3},
                                 {"patch_filter.min.0", 0},
                                 {"patch_filter.min.1", 0},
                                 {"patch_filter.extent.0", 7},
                                 {"patch_filter.extent.1", 7},
                                 {"search_area", 7},
                                 {"d_output.min.0", 0},
                                 {"d_output.min.1", 0},
                                 {"d_output.min.2", 0},
                                 {"d_output.extent.0", 512},
                                 {"d_output.extent.1", 512},
                                 {"d_output.extent.2", 3},
                                 },
                                {{{0, 512}, // d_input
                                  {0, 512},
                                  {0, 2}},
                                 {{0, 5},   // feature_filter
                                  {0, 5},
                                  {0, 2},
                                  {0, 2}},
                                 {{0, 7},   // patch_filter
                                  {0, 7}},
                                 {}         // inv_sigma
                                },
                                options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::NonLocalMeansBackwardGenerator, non_local_means_backward)

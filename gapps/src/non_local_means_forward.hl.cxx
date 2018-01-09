#include "algorithms/non_local_means.h"
#include "gradient_helpers.h"

namespace gradient_apps {

class NonLocalMeansForwardGenerator : public Generator<NonLocalMeansForwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 4};       // x, y, channel
    Input<Buffer<float>>  feature_filter{"feature_filter", 4};
    Input<Buffer<float>>  patch_filter{"patch_filter", 2};
    Input<float>          inv_sigma{"inv_sigma"};
    Input<int>            search_area{"search_area"};

    Output<Buffer<float>> output{"output", 4};     // x, y, channel

    void generate() {
        std::map<std::string, Func> func_map = non_local_means(
            input, feature_filter, patch_filter, inv_sigma, search_area);
        Func f_output = func_map["output"];
        output(x, y, c, n) = f_output(x, y, c, n);

        if(auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            Func output_func = output;
            simple_autoschedule(output_func,
                                {{"input.min.0", 0},
                                 {"input.min.1", 0},
                                 {"input.min.2", 0},
                                 {"input.min.3", 0},
                                 {"input.extent.0", 512},
                                 {"input.extent.1", 512},
                                 {"input.extent.2", 3},
                                 {"input.extent.4", 8},
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
                                 {"search_area", 7}
                                 },
                                {{0, 255},
                                 {0, 255},
                                 {0, 2},
                                 {0, 7}},
                                options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::NonLocalMeansForwardGenerator, non_local_means_forward)

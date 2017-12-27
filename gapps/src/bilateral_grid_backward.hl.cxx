#include "algorithms/bilateral_grid.h"
#include "gradient_helpers.h"

namespace gradient_apps {

class BilateralGridBackwardGenerator : public Generator<BilateralGridBackwardGenerator> {
public:
    Input<int> sigma_s{"sigma_s"}; // block_size in spatial
    Input<int> sigma_r{"sigma_r"}; // number of guide discrete levels

    Input<Buffer<float>>  input{"input", 3};       // x, y, channel
    Input<Buffer<float>>  filter_s{"filter_s", 1};
    Input<Buffer<float>>  filter_r{"filter_r", 1};
    Input<Buffer<float>>  d_output{"d_output", 3};

    Output<Buffer<float>> d_input{"d_input", 3};
    Output<Buffer<float>> d_filter_s{"d_filter_s", 1};
    Output<Buffer<float>> d_filter_r{"d_filter_r", 1};

    void generate() {
        std::map<std::string, Func> func_map = bilateral_grid(
            input, filter_s, filter_r, sigma_r);
        Func f_output = func_map["output"];
        Func f_input = func_map["f_input"];
        Func f_filter_s = func_map["f_filter_s"];
        Func f_filter_r = func_map["f_filter_r"];
        Derivative d = propagate_adjoints(
            f_output,
            d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()}}
        );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, f_input, d_input);
        assign_gradient(adjoints, f_filter_s, d_filter_s);
        assign_gradient(adjoints, f_filter_r, d_filter_r);

        if(auto_schedule) {
        } else {
            auto func_map = get_deps({d_input,
                                      d_filter_s,
                                      d_filter_r});
            std::vector<Func> funcs{d_input, d_filter_s, d_filter_r};
            simple_autoschedule(funcs,
                                {{"input.min.0", 0},
                                 {"input.min.1", 0},
                                 {"input.min.2", 0},
                                 {"input.extent.0", 256},
                                 {"input.extent.1", 256},
                                 {"input.extent.2", 3},
                                 {"filter_s.min.0", 0},
                                 {"filter_s.extent.0", 4},
                                 {"filter_r.min.0", 0},
                                 {"filter_r.extent.0", 4},
                                 {"d_output.min.0", 0},
                                 {"d_output.min.1", 0},
                                 {"d_output.min.2", 0},
                                 {"d_output.extent.0", 256},
                                 {"d_output.extent.1", 256},
                                 {"d_output.extent.2", 3},
                                 {"sigma_s", 4},
                                 {"sigma_r", 4}},
                                {{{0, 255},
                                  {0, 255},
                                  {0, 2}},
                                 {{0, 4}},
                                 {{0, 4}}});
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralGridBackwardGenerator, bilateral_grid_backward)

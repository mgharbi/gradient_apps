#include "algorithms/bilateral_slice_apply.h"
#include "gradient_helpers.h"

namespace gradient_apps {

class BilateralSliceApplyForwardGenerator : public Generator<BilateralSliceApplyForwardGenerator> {
public:
    Input<Buffer<float>>  grid{"grid", 5};   
    Input<Buffer<float>>  guide{"guide", 3};  
    Input<Buffer<float>>  input{"input", 4}; 
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        std::map<std::string, Func> func_map = bilateral_slice_apply(
            grid, guide, input);
        Func f_output = func_map["output"];
        output(x, y, c, n) = f_output(x, y, c, n);
        Func output_func = output;

        //Func affine_coeffs = func_map["affine_coeffs"];
        //affine_coeffs.memoize();

        if(auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            simple_autoschedule(output_func,
                                {
                                  {"grid.min.0", 0},
                                  {"grid.min.1", 0},
                                  {"grid.min.2", 0},
                                  {"grid.min.3", 0},
                                  {"grid.min.4", 0},
                                  {"grid.extent.0", 64},
                                  {"grid.extent.1", 64},
                                  {"grid.extent.2", 8},
                                  {"grid.extent.3", 12},
                                  {"grid.extent.4", 4},
                                  {"guide.min.0", 0},
                                  {"guide.min.1", 0},
                                  {"guide.min.2", 0},
                                  {"guide.extent.0", 2048},
                                  {"guide.extent.1", 2048},
                                  {"guide.extent.2", 4},
                                  {"input.min.0", 0},
                                  {"input.min.1", 0},
                                  {"input.min.2", 0},
                                  {"input.min.3", 0},
                                  {"input.extent.0", 2048},
                                  {"input.extent.1", 2048},
                                  {"input.extent.2", 3},
                                  {"input.extent.3", 4}
                                },
                                {{0, 2047}, {0, 2047}, {0, 2}, {0, 3}},
                                options,
                                {"affine_coeffs"});
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralSliceApplyForwardGenerator, bilateral_slice_apply_forward)

#include "algorithms/learnable_wiener.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class LearnableWienerBackwardGenerator
  : public Generator<LearnableWienerBackwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  reg_kernel{"reg_kernel", 2};
    Input<Buffer<float>>  d_output{"d_output", 3};

    Output<Buffer<float>>  d_reg_kernel{"d_reg_kernel", 2};

    void generate() {
        std::map<std::string, Func> func_map =
            learnable_wiener(320, 240, blurred, kernel, reg_kernel);
        Func filtered = func_map["filtered"];
        Func f_reg_kernel = func_map["reg_kernel"];

        Derivative d = propagate_adjoints(
            filtered, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()}
             });
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_reg_kernel.name(), -1}) != adjoints.end());
        Func d_reg_kernel_func = adjoints[FuncKey{f_reg_kernel.name(), -1}];
        d_reg_kernel(x, y) = d_reg_kernel_func(x, y);

        if (auto_schedule) {
            blurred.dim(0).set_bounds_estimate(0, 320);
            blurred.dim(1).set_bounds_estimate(0, 240);
            blurred.dim(2).set_bounds_estimate(0, 3);

            kernel.dim(0).set_bounds_estimate(0, 5);
            kernel.dim(1).set_bounds_estimate(0, 5);

            reg_kernel.dim(0).set_bounds_estimate(0, 320);
            reg_kernel.dim(1).set_bounds_estimate(0, 240);

            d_output.dim(0).set_bounds_estimate(0, 320);
            d_output.dim(1).set_bounds_estimate(0, 240);
            d_output.dim(2).set_bounds_estimate(0, 3);

            d_reg_kernel.estimate(x, 0, 320)
                        .estimate(y, 0, 240);
        } else {
            compute_all_root(d_reg_kernel_func);
            std::cerr << "rooted" << std::endl;
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::LearnableWienerBackwardGenerator, learnable_wiener_backward)

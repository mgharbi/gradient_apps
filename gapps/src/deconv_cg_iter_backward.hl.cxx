#include "gradient_helpers.h"

#include "algorithms/deconv_cg_iter.h"

namespace gradient_apps {

class DeconvCgIterBackwardGenerator
  : public Generator<DeconvCgIterBackwardGenerator> {
public:
    Input<Buffer<float>>  xrp{"xrp", 4};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernel", 3};
    Input<Buffer<float>>  d_next_xrp{"d_next_xrp", 4};
    Output<Buffer<float>> d_xrp{"d_xrp", 4};
    Output<Buffer<float>> d_reg_kernel_weights{"d_reg_kernel_weights", 1};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernel", 3};

    void generate() {
        auto func_map = deconv_cg_iter(xrp, kernel, reg_kernel_weights, reg_kernels);
        Func xrp_func = func_map["xrp_func"];
        Func reg_kernel_weights_func = func_map["reg_kernel_weights_func"];
        Func reg_kernels_func = func_map["reg_kernels_func"];
        Func next_xrp = func_map["next_xrp"];
        Derivative d = propagate_adjoints(
            next_xrp,
            d_next_xrp,
            {{d_next_xrp.dim(0).min(), d_next_xrp.dim(0).max()},
             {d_next_xrp.dim(1).min(), d_next_xrp.dim(1).max()},
             {d_next_xrp.dim(2).min(), d_next_xrp.dim(2).max()},
             {d_next_xrp.dim(3).min(), d_next_xrp.dim(3).max()}}
        );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{xrp_func.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{reg_kernel_weights_func.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{reg_kernels_func.name(), -1}) != adjoints.end());
        d_xrp(x, y, c, n) = adjoints[FuncKey{xrp_func.name(), -1}](x, y, c, n);
        d_reg_kernel_weights(n) = adjoints[FuncKey{reg_kernel_weights_func.name(), -1}](n);
        d_reg_kernels(x, y, n) = adjoints[FuncKey{reg_kernels_func.name(), -1}](x, y, n);

        if (auto_schedule) {
            xrp.dim(0).set_bounds_estimate(0, 320);
            xrp.dim(1).set_bounds_estimate(0, 240);
            xrp.dim(2).set_bounds_estimate(0, 3);
            xrp.dim(3).set_bounds_estimate(0, 3);

            kernel.dim(0).set_bounds_estimate(0, 5);
            kernel.dim(1).set_bounds_estimate(0, 5);

            reg_kernel_weights.dim(0).set_bounds_estimate(0, 2);

            reg_kernels.dim(0).set_bounds_estimate(0, 3);
            reg_kernels.dim(1).set_bounds_estimate(0, 3);
            reg_kernels.dim(2).set_bounds_estimate(0, 2);

            d_next_xrp.dim(0).set_bounds_estimate(0, 320);
            d_next_xrp.dim(1).set_bounds_estimate(0, 240);
            d_next_xrp.dim(2).set_bounds_estimate(0, 3);
            d_next_xrp.dim(3).set_bounds_estimate(0, 3);

            d_xrp.estimate(x, 0, 320)
                 .estimate(y, 0, 240)
                 .estimate(c, 0, 3)
                 .estimate(n, 0, 3);

            d_reg_kernel_weights.estimate(n, 0, 2);
            d_reg_kernels.estimate(x, 0, 3)
                         .estimate(y, 0, 3)
                         .estimate(n, 0, 2);
        } else {
            compute_all_root(d_xrp);
            compute_all_root(d_reg_kernel_weights);
            compute_all_root(d_reg_kernels);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgIterBackwardGenerator, deconv_cg_iter_backward)


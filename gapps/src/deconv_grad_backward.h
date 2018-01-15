#include "gradient_helpers.h"

#include "algorithms/deconv_cg.h"

namespace gradient_apps {

class DeconvGradBackwardGenerator
  : public Generator<DeconvGradBackwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  xk{"xk", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  data_kernel_weights{"data_kernel_weights", 1};
    Input<Buffer<float>>  data_kernels{"data_kernels", 3};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  reg_targets{"reg_targets", 4};
#ifdef NEED_HESS
    Input<Buffer<float>>  hess_dir{"hess_dir", 3};
    Input<Buffer<float>>  d_output{"d_output", 4};
#else
    Input<Buffer<float>>  d_output{"d_output", 3};
#endif
    Output<Buffer<float>> d_xk{"d_xk", 3};
    Output<Buffer<float>> d_data_kernel_weights{"d_data_kernel_weights", 1};
    Output<Buffer<float>> d_data_kernels{"d_data_kernels", 3};
    Output<Buffer<float>> d_reg_kernel_weights{"d_reg_kernel_weights", 1};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernels", 3};
    Output<Buffer<float>> d_reg_targets{"d_reg_targets", 4};
#ifdef NEED_HESS
    Output<Buffer<float>> d_hess_dir{"d_hess_dir", 3};
#endif

    void generate() {
        Func grad = deconv_grad(
            xk, blurred, kernel,
            data_kernel_weights, data_kernels,
            reg_kernel_weights, reg_kernels, reg_targets);

        Func output("output");
#ifdef NEED_HESS
        // Use forward autodiff to get Hessian-vector product
        Func hess = propagate_tangents(grad, {{xk.name(), Func(hess_dir)}});
        output(x, y, c, n) = 0.f;
        output(x, y, c, 0) = grad(x, y, c);
        output(x, y, c, 1) = hess(x, y, c);
        Derivative d = propagate_adjoints(
            output,
            d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}}
        );
#else
        output(x, y, c) = grad(x, y, c);
        Derivative d = propagate_adjoints(
            output,
            d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()}}
        );
#endif
        assign_gradient(d, xk, d_xk);
        assign_gradient(d, data_kernel_weights, d_data_kernel_weights);
        assign_gradient(d, data_kernels, d_data_kernels);
        assign_gradient(d, reg_kernel_weights, d_reg_kernel_weights);
        assign_gradient(d, reg_kernels, d_reg_kernels);
        assign_gradient(d, reg_targets, d_reg_targets);
#ifdef NEED_HESS
        assign_gradient(d, hess_dir, d_hess_dir);
#endif

        if (auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
#ifdef NEED_HESS
            std::vector<Func> funcs{d_xk,
                d_data_kernel_weights, d_data_kernels,
                d_reg_kernel_weights, d_reg_kernels, d_reg_targets,
                d_hess_dir};
#else
            std::vector<Func> funcs{d_xk,
                d_data_kernel_weights, d_data_kernels,
                d_reg_kernel_weights, d_reg_kernels, d_reg_targets};
#endif
            simple_autoschedule(funcs,
                                {
                                 {"blurred.min.0", 0},
                                 {"blurred.min.1", 0},
                                 {"blurred.min.2", 0},
                                 {"blurred.extent.0", 256},
                                 {"blurred.extent.1", 256},
                                 {"blurred.extent.2", 3},
                                 {"xk.min.0", 0},
                                 {"xk.min.1", 0},
                                 {"xk.min.2", 0},
                                 {"xk.extent.0", 256},
                                 {"xk.extent.1", 256},
                                 {"xk.extent.2", 3},
                                 {"kernel.min.0", 0},
                                 {"kernel.min.1", 0},
                                 {"kernel.extent.0", 11},
                                 {"kernel.extent.1", 11},
                                 {"data_kernel_weights.min.0", 0},
                                 {"data_kernel_weights.extent.0", 5},
                                 {"data_kernels.min.0", 0},
                                 {"data_kernels.min.1", 0},
                                 {"data_kernels.min.2", 0},
                                 {"data_kernels.extent.0", 5},
                                 {"data_kernels.extent.1", 5},
                                 {"data_kernels.extent.2", 5},
                                 {"reg_kernel_weights.min.0", 0},
                                 {"reg_kernel_weights.extent.0", 5},
                                 {"reg_kernels.min.0", 0},
                                 {"reg_kernels.min.1", 0},
                                 {"reg_kernels.min.2", 0},
                                 {"reg_kernels.extent.0", 5},
                                 {"reg_kernels.extent.1", 5},
                                 {"reg_kernels.extent.2", 5},
                                 {"reg_targets.min.0", 0},
                                 {"reg_targets.min.1", 0},
                                 {"reg_targets.min.2", 0},
                                 {"reg_targets.min.3", 0},
                                 {"reg_targets.extent.0", 256},
                                 {"reg_targets.extent.1", 256},
                                 {"reg_targets.extent.2", 3},
                                 {"reg_targets.extent.3", 5},
                                 {"d_output.min.0", 0},
                                 {"d_output.min.1", 0},
                                 {"d_output.min.2", 0},
                                 {"d_output.extent.0", 256},
                                 {"d_output.extent.1", 256},
                                 {"d_output.extent.2", 3},
#ifdef NEED_HESS
                                 {"d_output.min.3", 0},
                                 {"d_output.extent.3", 2},
#endif
                                },
                                {
                                 {{0, 255}, // xk
                                  {0, 255},
                                  {0, 2}},
                                 {{0, 4}}, // data_kernel_weights
                                 {{0, 4},  // data_kernels
                                  {0, 4},
                                  {0, 4}},
                                 {{0, 4}}, // reg_kernel_weights
                                 {{0, 4},  // reg_kernels
                                  {0, 4},
                                  {0, 4}},
                                 {{0, 255}, // r_targets
                                  {0, 255},
                                  {0, 2},
                                  {0, 4}},
#ifdef NEED_HESS
                                 {{0, 255}, // hess_dir
                                  {0, 255},
                                  {0, 2}},
#endif
                                },
                                options,
                                {"output_1_d_def__$1"});
        }
    }
};

}  // end namespace gradient_apps


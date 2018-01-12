#include "gradient_helpers.h"

#include "algorithms/deconv_cg.h"

namespace gradient_apps {

class DeconvGradForwardGenerator
  : public Generator<DeconvGradForwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  xk{"xk", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  data_kernel_weights{"data_kernel_weights", 1};
    Input<Buffer<float>>  data_kernels{"data_kernels", 3};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  reg_targets{"reg_targets", 4};
    Input<Buffer<float>>  hess_dir{"hess_dir", 3};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        // Boundary condition
        //Func blurred_re, clamped_blurred;
        //std::tie(blurred_re, clamped_blurred) =
        //    select_repeat_edge(blurred, blurred.width(), blurred.height());
        //Func xk_re, clamped_xk;
        //std::tie(xk_re, clamped_xk) = select_repeat_edge(xk, xk.width(), xk.height());
        //RDom r_image(0, xk.width(), 0, xk.height(), 0, xk.channels());
        //Func grad = deconv_grad(
        //    xk, blurred, kernel,
        //    data_kernel_weights, data_kernels,
        //    reg_kernel_weights, reg_kernels, reg_targets);
        RDom r_kernel(kernel);
        RDom r_image(0, xk.width(), 0, xk.height(), 0, xk.channels());
        Func clamped_xk = BoundaryConditions::repeat_edge(xk);
        // Define cost function
        // data term
        Func kx("kx");
        kx(x, y, c) = 0.f;
        kx(x, y, c) += clamped_xk(x + r_kernel.x - kernel.width()  / 2,
                                  y + r_kernel.y - kernel.height() / 2,
                                  c) *
                       kernel(r_kernel.x, r_kernel.y);

        Func data_term("data_term");
        data_term() = 0.f;
        data_term() += pow(kx(r_image.x, r_image.y, r_image.z) -
                           blurred(r_image.x, r_image.y, r_image.z), 2.f);

        Derivative d = propagate_adjoints(data_term);
        Func grad = d(xk);

        // Use forward autodiff to get Hessian-vector product
        // Generate two versions of the function:
        // First version uses Hessian-gradient product
        // Second version uses Hessian-hess_dir product
#if INIT
        Func hess = propagate_tangents(grad, {{xk.name(), grad}});
#else
        Func hess = propagate_tangents(grad, {{xk.name(), Func(hess_dir)}});
#endif
        output(x, y, c, n) = 0.f;
        output(x, y, c, 0) = grad(x, y, c);
        output(x, y, c, 1) = hess(x, y, c);

        if (auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            Func output_func = output;
            simple_autoschedule(output_func,
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
                                },
                                {
                                 {{0, 255},
                                  {0, 255},
                                  {0, 2},
                                  {0, 1}},
                                },
                                options);
        }
    }
};

}  // end namespace gradient_apps


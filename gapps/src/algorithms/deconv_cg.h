#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
Func deconv_grad(const Input &xk,
                 const Input &blurred,
                 //const RDom  &r_image,
                 const Input &kernel,
                 const Input &data_kernel_weights,
                 const Input &data_kernels,
                 const Input &reg_kernel_weights,
                 const Input &reg_kernels,
                 const Input &rtargets) {
    RDom r_kernel(kernel);
    RDom r_data_kernel_xy(0, data_kernels.width(), 0, data_kernels.height());
    RDom r_data_kernel_z(0, data_kernels.channels());
    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    RDom r_image(0, xk.width(), 0, xk.height(), 0, xk.channels());

    Func clamped_xk = BoundaryConditions::repeat_edge(xk);
    Func clamped_blurred = BoundaryConditions::repeat_edge(blurred);
    // Define cost function
    // data term
    Func kx("kx");
    kx(x, y, c) = 0.f;
    kx(x, y, c) += clamped_xk(x + r_kernel.x - kernel.width()  / 2,
                              y + r_kernel.y - kernel.height() / 2,
                              c) *
                   kernel(r_kernel.x, r_kernel.y);

    Func dkx("dkx");
    dkx(x, y, c, n) = 0.f;
    dkx(x, y, c, n) += kx(x + r_data_kernel_xy.x - data_kernels.width()  / 2,
                          y + r_data_kernel_xy.y - data_kernels.height() / 2,
                          c) *
                       data_kernels(r_data_kernel_xy.x, r_data_kernel_xy.y, n);
    // TODO: this term is recomputed across CG iterations, should be precomputed
    Func dki("dki");
    dki(x, y, c, n) = 0.f;
    dki(x, y, c, n) += clamped_blurred(x + r_data_kernel_xy.x - data_kernels.width()  / 2,
                                       y + r_data_kernel_xy.y - data_kernels.height() / 2,
                                       c) *
                       data_kernels(r_data_kernel_xy.x, r_data_kernel_xy.y, n);
    Func data_term("data_term");
    data_term(n) = 0.f;
    data_term(n) += pow(dkx(r_image.x, r_image.y, r_image.z, n) -
                        dki(r_image.x, r_image.y, r_image.z, n), 2.f) *
                    abs(data_kernel_weights(n));

    // regularization term
    Func rkx("rkx");
    rkx(x, y, c, n) = 0.f;
    rkx(x, y, c, n) += clamped_xk(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                  y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                  c) *
                       reg_kernels(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func reg_term("reg_term");
    reg_term(n) = 0.f;
    reg_term(n) += pow(rkx(r_image.x, r_image.y, r_image.z, n) -
                       rtargets(r_image.x, r_image.y, r_image.z, n), 2.f) *
                   abs(reg_kernel_weights(n));
    Func loss("loss");
    loss() = 0.f;
    loss() += data_term(r_data_kernel_z);
    loss() += reg_term(r_reg_kernel_z);

    // Use autodiff to get gradient
    Derivative d = propagate_adjoints(loss);
    return d(xk);
}

// The Code below is not used, just for a reference how CG would look like in Halide
//
template <typename Input>
std::map<std::string, Func> deconv_cg_init(const Input &blurred,
                                           const Input &x0,
                                           const Input &kernel,
                                           const Input &data_kernel_weights,
                                           const Input &data_kernels,
                                           const Input &reg_kernel_weights,
                                           const Input &reg_kernels,
                                           const Input &rtargets) {
    // Boundary condition
    Func blurred_re, clamped_blurred;
    std::tie(blurred_re, clamped_blurred) = select_repeat_edge(blurred, blurred.width(), blurred.height());
    Func x0_re, clamped_x0;
    std::tie(x0_re, clamped_x0) = select_repeat_edge(x0, x0.width(), x0.height());
    RDom r_image(0, x0.width(), 0, x0.height(), 0, x0.channels());
    Func grad = deconv_grad(clamped_x0,
                            clamped_blurred,
                            r_image,
                            kernel,
                            data_kernel_weights,
                            data_kernels,
                            reg_kernel_weights,
                            reg_kernels,
                            rtargets);
    Func r("r"); // steepest descent direction
    r(x, y, c) = -grad(x, y, c);
    Func p("p"); // conjugate direction
    p(x, y, c) = r(x, y, c);
    // Use forward autodiff to get Hessian-vector product
    Func hess_p = propagate_tangents(grad, {{x0.name(), p}});
    Func xrph("xrph");
    xrph(x, y, c, n) = 0.f;
    xrph(x, y, c, 0) = clamped_x0(x, y, c);
    xrph(x, y, c, 1) = r(x, y, c);
    xrph(x, y, c, 2) = p(x, y, c);
    xrph(x, y, c, 3) = hess_p(x, y, c);
    return {{"x0", x0_re}, {"xrph", xrph}};
}

template <typename Input>
std::map<std::string, Func> deconv_cg_iter(const Input &blurred,
                                           const Input &xrph,
                                           const Input &kernel,
                                           const Input &data_kernel_weights,
                                           const Input &data_kernels,
                                           const Input &reg_kernel_weights,
                                           const Input &reg_kernels,
                                           const Input &rtargets) {
    // Boundary condition
    Func blurred_re, clamped_blurred;
    std::tie(blurred_re, clamped_blurred) = select_repeat_edge(blurred, blurred.width(), blurred.height());
    Func xrph_re, clamped_xrph;
    std::tie(xrph_re, clamped_xrph) = select_repeat_edge(xrph, xrph.width(), xrph.height());
    // Extract xrp
    Func xk("xk");
    xk(x, y, c) = clamped_xrph(x, y, c, 0);
    Func r("r");
    r(x, y, c) = clamped_xrph(x, y, c, 1);
    Func p("p");
    p(x, y, c) = clamped_xrph(x, y, c, 2);
    Func h("h");
    h(x, y, c) = clamped_xrph(x, y, c, 3);
    RDom r_image(0, xrph.width(), 0, xrph.height(), 0, xrph.channels());

    // One step line search
    // alpha = r^T * p / (p^T hess p)
    Func grad_dot_p("grad_dot_p");
    grad_dot_p() = 0.f;
    grad_dot_p() += r(r_image.x, r_image.y, r_image.z) *
                    p(r_image.x, r_image.y, r_image.z);
    Func p_dot_hess_p("p_dot_hess_p");
    p_dot_hess_p() = 0.f;
    p_dot_hess_p() += p(r_image.x, r_image.y, r_image.z) *
                      h(r_image.x, r_image.y, r_image.z);
    Func alpha("alpha");
    alpha() = grad_dot_p() / max(p_dot_hess_p(), 1e-8f);
    Func next_x("next_x");
    next_x(x, y, c) = xk(x, y, c) + alpha() * p(x, y, c);
    Func grad = deconv_grad(next_x,
                            clamped_blurred,
                            r_image,
                            kernel,
                            data_kernel_weights,
                            data_kernels,
                            reg_kernel_weights,
                            reg_kernels,
                            rtargets);
    Func next_r("next_r");
    next_r(x, y, c) = -grad(x, y, c);

    // beta = next_r^T next_r / r^T r
    // Use Fletcher-Reeves update rule
    Func grad_norm("grad_norm");
    grad_norm() = 0.f;
    grad_norm() += grad(r_image.x, r_image.y, r_image.z) *
                   grad(r_image.x, r_image.y, r_image.z);
    Func prev_grad_norm("prev_grad_norm");
    prev_grad_norm() = 0.f;
    prev_grad_norm() += r(r_image.x, r_image.y, r_image.z) *
                        r(r_image.x, r_image.y, r_image.z);
    Func beta("beta");
    beta() = grad_norm() / prev_grad_norm();

    Func next_p("next_p");
    next_p(x, y, c) = -grad(x, y, c) + beta() * p(x, y, c);
    // Use forward autodiff to get Hessian-vector product
    Func hess_p = propagate_tangents(grad, {{next_x.name(), next_p}});

    Func next_xrph("next_xrph");
    next_xrph(x, y, c, n) = 0.f;
    next_xrph(x, y, c, 0) = next_x(x, y, c);
    next_xrph(x, y, c, 1) = next_r(x, y, c);
    next_xrph(x, y, c, 2) = next_p(x, y, c);
    next_xrph(x, y, c, 3) = hess_p(x, y, c);
    return {{"xrph", xrph_re}, {"next_xrph", next_xrph}};
}


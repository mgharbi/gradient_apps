#pragma once

#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
Func deconv_grad(const Input &xk,
                 const Input &blurred,
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
    // TODO: this term is recomputed across CG iterations, should be precomputed (memonize?)
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


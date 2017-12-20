#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> deconv_cg_iter(
        const Input &xrp,
        const Input &kernel,
        const Input &reg_kernel_weights,
        const Input &reg_kernels,
	const Input &precond_kernel,
        const Input &w_kernel,
        const Input &w_reg_kernels) {
    // A single iteration of conjugate gradient, takes X, R, P and updates them
    Func xrp_func("xrp_func");
    xrp_func(x, y, c, n) = xrp(x, y, c, n);
    Func reg_kernel_weights_func("reg_kernel_weights_func");
    reg_kernel_weights_func(n) = reg_kernel_weights(n);
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    Func precond_kernel_func("precond_kernel_func");
    precond_kernel_func(x, y) = precond_kernel(x, y);
    Func w_kernel_func("w_kernel_func");
    w_kernel_func(x, y, c) = w_kernel(x, y, c);
    Func w_reg_kernels_func("w_reg_kernels_func");
    w_reg_kernels_func(x, y, c, n) = w_reg_kernels(x, y, c, n);

    RDom r_image(0, xrp.width(), 0, xrp.height(), 0, xrp.channels());
    RDom r_kernel(kernel);
    Func xrp_clamped = BoundaryConditions::repeat_edge(xrp_func,
                {{Expr(0), Expr(xrp.width())},
                 {Expr(0), Expr(xrp.height())},
                 {Expr(), Expr()},
                 {Expr(), Expr()}});
    Func clamped_w_kernel = BoundaryConditions::repeat_edge(w_kernel_func,
                {{Expr(0), Expr(xrp.width())},
                 {Expr(0), Expr(xrp.height())},
                 {Expr(), Expr()}});
    Func clamped_w_reg_kernels = BoundaryConditions::repeat_edge(w_reg_kernels_func,
                {{Expr(0), Expr(xrp.width())},
                 {Expr(0), Expr(xrp.height())},
                 {Expr(), Expr()},
                 {Expr(), Expr()}});
    // Extract input
    Func xk("xk");
    xk(x, y, c) = xrp_clamped(x, y, c, 0);
    Func r("r");
    r(x, y, c) = xrp_clamped(x, y, c, 1);
    Func p("p");
    p(x, y, c) = xrp_clamped(x, y, c, 2);
    Func z("z");
    z(x, y, c) = xrp_clamped(x, y, c, 3);

    Func rTz("rTz");
    // alpha = r^T * z / p^T A^T W A p
    rTz() = 0.f;
    rTz() += r(r_image.x, r_image.y, r_image.z) *
             z(r_image.x, r_image.y, r_image.z);
    Func Kp("Kp");
    Kp(x, y, c) = 0.f;
    Kp(x, y, c) += p(x + r_kernel.x - kernel.width()  / 2,
                     y + r_kernel.y - kernel.height() / 2,
                     c) *
                   kernel(r_kernel.x, r_kernel.y);
    Func WKp("WKp");
    WKp(x, y, c) = Kp(x, y, c) * clamped_w_kernel(x, y, c);
    Func KTWKp("K^TWKp");
    KTWKp(x, y, c) = 0.f;
    KTWKp(x, y, c) += WKp(x + r_kernel.x - kernel.width()  / 2,
                          y + r_kernel.y - kernel.height() / 2,
                          c) *
                       kernel(kernel.width()  - r_kernel.x - 1,
                              kernel.height() - r_kernel.y - 1);
    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func rKp("rKp");
    rKp(x, y, c, n) = 0.f;
    rKp(x, y, c, n) += p(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                         y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                         c) *
                       reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func WrKp("WrKp");
    WrKp(x, y, c, n) = rKp(x, y, c, n) * clamped_w_reg_kernels(x, y, c, n);
    Func rKTWrKp("rK^TWrKp");
    rKTWrKp(x, y, c, n) = 0.f;
    rKTWrKp(x, y, c, n) += WrKp(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                c,
                                n) *
                           reg_kernels_func(reg_kernels.width()  - r_reg_kernel_xy.x - 1,
                                            reg_kernels.height() - r_reg_kernel_xy.y - 1,
                                            n);
    Func ATWAp("A^TWAp");
    ATWAp(x, y, c) = KTWKp(x, y, c);
    ATWAp(x, y, c) += rKTWrKp(x, y, c, r_reg_kernel_z.x) *
                      reg_kernel_weights_func(r_reg_kernel_z.x) *
                      reg_kernel_weights_func(r_reg_kernel_z.x);
    Func pTATWAp("p^TA^TWAp");
    pTATWAp() = 0.f;
    pTATWAp() += p(r_image.x, r_image.y, r_image.z) *
                 ATWAp(r_image.x, r_image.y, r_image.z);

    Func alpha("alpha");
    alpha() = rTz() / pTATWAp();
    // x = x + alpha * p
    Func next_x("next_x");
    next_x(x, y, c) = xk(x, y, c) + alpha() * p(x, y, c);
    // r = r - alpha * A^TAp
    Func next_r("next_r");
    next_r(x, y, c) = r(x, y, c) - alpha() * ATWAp(x, y, c);

    // beta = nextR^TnextR / r^Tr
    Func nRTnR("nRTnR");
    nRTnR() = 0.f;
    nRTnR() += next_r(r_image.x, r_image.y, r_image.z) *
               next_r(r_image.x, r_image.y, r_image.z);
    Func beta("beta");
    beta() = nRTnR() / rTr();
    Func next_p("next_p");
    next_p(x, y, c) = next_r(x, y, c) + beta() * p(x, y, c);

    Func next_xrp("next_xrp");
    next_xrp(x, y, c, n) = 0.f;
    next_xrp(x, y, c, 0) = next_x(x, y, c);
    next_xrp(x, y, c, 1) = next_r(x, y, c);
    next_xrp(x, y, c, 2) = next_p(x, y, c);

    std::map<std::string, Func> func_map;
    func_map["reg_kernel_weights_func"] = reg_kernel_weights_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["w_kernel_func"] = w_kernel_func;
    func_map["w_reg_kernels_func"] = w_reg_kernels_func;
    func_map["xrp_func"] = xrp_func;
    func_map["next_xrp"] = next_xrp;
    return func_map;
}


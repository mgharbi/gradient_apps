#pragma once

#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

Expr square(Expr e) {
    return e * e;
}

template <typename Input>
Func deconv_cg_iter(
        const Input &xrp,
        const Input &kernel,
        const Input &data_kernel_weights,
        const Input &data_kernels,
        const Input &reg_kernel_weights,
        const Input &reg_kernels,
        const Input &w_data,
        const Input &w_reg) {
    // A single iteration of conjugate gradient, takes X, R, P and updates them
    Expr kw = kernel.width();
    Expr kh = kernel.height();
    Expr dkw = data_kernels.width();
    Expr dkh = data_kernels.height();
    Expr rkw = reg_kernels.width();
    Expr rkh = reg_kernels.height();
    RDom r_k(kernel);
    RDom r_dk(0, dkw, 0, dkh);
    RDom r_dk_c(0, data_kernels.channels());
    RDom r_rk(0, rkw, 0, rkh);
    RDom r_rk_c(0, reg_kernels.channels());
    RDom r_img(0, xrp.width(), 0, xrp.height(), 0, xrp.channels());
    Func clamped_xrp = BoundaryConditions::repeat_edge(xrp);
    Func clamped_w_data = BoundaryConditions::repeat_edge(w_data);
    Func clamped_w_reg = BoundaryConditions::repeat_edge(w_reg);

    // Extract input
    Func xk("xk");
    xk(x, y, c) = clamped_xrp(x, y, c, 0);
    Func r("r");
    r(x, y, c) = clamped_xrp(x, y, c, 1);
    Func p("p");
    p(x, y, c) = clamped_xrp(x, y, c, 2);
    Func z("z");
    z(x, y, c) = clamped_xrp(x, y, c, 3);

    Func rTr("rTr");
    // alpha = r^T * r / p^T A^T W A p
    rTr() = sum(square(r(r_img.x, r_img.y, r_img.z)));
    // Data term on p
    Func Kp("Kp");
    Kp(x, y, c) = sum(p(x + r_k.x - kw / 2, y + r_k.y - kh / 2, c) *
                      kernel(r_k.x, r_k.y));
    Func dKp("dKp");
    dKp(x, y, c, n) = sum(Kp(x + r_dk.x - dkw / 2, y + r_dk.y - dkh / 2, c) *
                          data_kernels(r_dk.x, r_dk.y, n));
    Func WdKp("WdKp");
    WdKp(x, y, c, n) = dKp(x, y, c, n) * clamped_w_data(x, y, c, n);
    Func dKTWdKp("dK^TWdKp");
    dKTWdKp(x, y, c, n) = sum(WdKp(x - r_dk.x + dkw / 2, y - r_dk.y + dkh / 2, c, n) *
                              data_kernels(r_dk.x, r_dk.y, n));
    Func wdKTWdKp("wdKTWdKp");
    wdKTWdKp(x, y, c) = sum(dKTWdKp(x, y, c, r_dk_c) *
                            data_kernel_weights(r_dk_c));
    Func KTWKp("K^TWKp");
    KTWKp(x, y, c) = sum(wdKTWdKp(x - r_k.x + kw / 2, y - r_k.y + kh / 2, c) *
                         kernel(r_k.x, r_k.y));
    // Prior term on p
    Func rKp("rKp");
    rKp(x, y, c, n) = sum(p(x + r_rk.x - rkw / 2, y + r_rk.y - rkh / 2, c) *
                          reg_kernels(r_rk.x, r_rk.y, n));
    Func WrKp("WrKp");
    WrKp(x, y, c, n) = rKp(x, y, c, n) * clamped_w_reg(x, y, c, n);
    Func rKTWrKp("rK^TWrKp");
    rKTWrKp(x, y, c, n) = 0.f;
    rKTWrKp(x, y, c, n) += WrKp(x - r_rk.x + rkw / 2, y - r_rk.y + rkh / 2, c, n) *
                           reg_kernels(r_rk.x, r_rk.y, n);
    Func wrKTWrKp("wrKTWrKp");
    wrKTWrKp(x, y, c) = 0.f;
    wrKTWrKp(x, y, c) += rKTWrKp(x, y, c, r_rk_c) *
                         abs(reg_kernel_weights(r_rk_c));
    Func ATWAp("A^TWAp");
    ATWAp(x, y, c) = KTWKp(x, y, c) + wrKTWrKp(x, y, c);
    Func pTATWAp("p^TA^TWAp");
    pTATWAp() = 0.f;
    pTATWAp() += p(r_img.x, r_img.y, r_img.z) *
                 ATWAp(r_img.x, r_img.y, r_img.z);

    Func alpha("alpha");
    alpha() = rTr() / pTATWAp();
    // x = x + alpha * p
    Func next_x("next_x");
    next_x(x, y, c) = xk(x, y, c) + alpha() * p(x, y, c);
    // r = r - alpha * A^TAp
    Func next_r("next_r");
    next_r(x, y, c) = r(x, y, c) - alpha() * ATWAp(x, y, c);

    // beta = nextZ^T(nextR - r) / r^Tr
    Func nRTnR("nRTnR");
    nRTnR() = sum(square(next_r(r_img.x, r_img.y, r_img.z)));
    Func beta("beta");
    beta() = nRTnR() / rTr();
    Func next_p("next_p");
    next_p(x, y, c) = next_r(x, y, c) + beta() * p(x, y, c);

    Func next_xrp("next_xrp");
    next_xrp(x, y, c, n) = 0.f;
    next_xrp(x, y, c, 0) = next_x(x, y, c);
    next_xrp(x, y, c, 1) = next_r(x, y, c);
    next_xrp(x, y, c, 2) = next_p(x, y, c);

    return next_xrp;
}


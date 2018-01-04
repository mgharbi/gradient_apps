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
        const Input &data_kernel_weights,
        const Input &data_kernels,
        const Input &reg_kernel_weights,
        const Input &reg_kernels,
        const Input &w_data,
        const Input &w_reg) {
    // A single iteration of conjugate gradient, takes X, R, P, Z and updates them
    Func xrp_func("xrp_func");
    xrp_func(x, y, c, n) = xrp(x, y, c, n);
    Func data_kernel_weights_func("data_kernel_weights_func");
    data_kernel_weights_func(n) = data_kernel_weights(n);
    Func data_kernels_func("data_kernels_func");
    data_kernels_func(x, y, n) = data_kernels(x, y, n);
    Func reg_kernel_weights_func("reg_kernel_weights_func");
    reg_kernel_weights_func(n) = reg_kernel_weights(n);
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    Func w_data_func("w_data_func");
    w_data_func(x, y, c, n) = w_data(x, y, c, n);
    Func w_reg_func("w_reg_func");
    w_reg_func(x, y, c, n) = w_reg(x, y, c, n);

    RDom r_image(0, xrp.width(), 0, xrp.height(), 0, xrp.channels());
    RDom r_kernel(kernel);
    RDom r_data_kernel_xy(0, data_kernels.width(), 0, data_kernels.height());
    RDom r_data_kernel_z(0, data_kernels.channels());
    Func xrp_re, clamped_xrp;
    std::tie(xrp_re, clamped_xrp) = select_repeat_edge(xrp_func, xrp.width(), xrp.height());
    Func w_data_re, clamped_w_data;
    std::tie(w_data_re, clamped_w_data) = select_repeat_edge(w_data_func, xrp.width(), xrp.height());
    Func w_reg_re, clamped_w_reg;
    std::tie(w_reg_re, clamped_w_reg) = select_repeat_edge(w_reg_func, xrp.width(), xrp.height());
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
    rTr() = 0.f;
    rTr() += r(r_image.x, r_image.y, r_image.z) *
             r(r_image.x, r_image.y, r_image.z);
    // Data term on p
    Func Kp("Kp");
    Kp(x, y, c) = 0.f;
    Kp(x, y, c) += p(x + r_kernel.x - kernel.width()  / 2,
                     y + r_kernel.y - kernel.height() / 2,
                     c) *
                   kernel(r_kernel.x, r_kernel.y);
    Func dKp("dKp");
    dKp(x, y, c, n) = 0.f;
    dKp(x, y, c, n) += Kp(x + r_data_kernel_xy.x - data_kernels.width()  / 2,
                          y + r_data_kernel_xy.y - data_kernels.height() / 2,
                          c) *
                       data_kernels_func(r_data_kernel_xy.x, r_data_kernel_xy.y, n);
    Func WdKp("WdKp");
    WdKp(x, y, c, n) = dKp(x, y, c, n) * clamped_w_data(x, y, c, n);
    Func dKTWdKp("dK^TWdKp");
    dKTWdKp(x, y, c, n) = 0.f;
    dKTWdKp(x, y, c, n) += WdKp(x - r_data_kernel_xy.x + data_kernels.width()  / 2,
                                y - r_data_kernel_xy.y + data_kernels.height() / 2,
                                c, n) *
                           data_kernels_func(r_data_kernel_xy.x, r_data_kernel_xy.y, n);
    Func wdKTWdKp("wdKTWdKp");
    wdKTWdKp(x, y, c) = 0.f;
    wdKTWdKp(x, y, c) += dKTWdKp(x, y, c, r_data_kernel_z) *
                         data_kernel_weights_func(r_data_kernel_z);
    Func KTWKp("K^TWKp");
    KTWKp(x, y, c) = 0.f;
    KTWKp(x, y, c) += wdKTWdKp(x - r_kernel.x + kernel.width()  / 2,
                               y - r_kernel.y + kernel.height() / 2,
                               c) *
                      kernel(r_kernel.x, r_kernel.y);
    // Prior term on p
    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func rKp("rKp");
    rKp(x, y, c, n) = 0.f;
    rKp(x, y, c, n) += p(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                         y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                         c) *
                       reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func WrKp("WrKp");
    WrKp(x, y, c, n) = rKp(x, y, c, n) * clamped_w_reg(x, y, c, n);
    Func rKTWrKp("rK^TWrKp");
    rKTWrKp(x, y, c, n) = 0.f;
    rKTWrKp(x, y, c, n) += WrKp(x - r_reg_kernel_xy.x + reg_kernels.width()  / 2,
                                y - r_reg_kernel_xy.y + reg_kernels.height() / 2,
                                c,
                                n) *
                           reg_kernels_func(r_reg_kernel_xy.x,
                                            r_reg_kernel_xy.y,
                                            n);
    Func wrKTWrKp("wrKTWrKp");
    wrKTWrKp(x, y, c) = 0.f;
    wrKTWrKp(x, y, c) += rKTWrKp(x, y, c, r_reg_kernel_z) *
                         abs(reg_kernel_weights_func(r_reg_kernel_z));
    Func ATWAp("A^TWAp");
    ATWAp(x, y, c) = KTWKp(x, y, c) + wrKTWrKp(x, y, c);
    Func pTATWAp("p^TA^TWAp");
    pTATWAp() = 0.f;
    pTATWAp() += p(r_image.x, r_image.y, r_image.z) *
                 ATWAp(r_image.x, r_image.y, r_image.z);

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
    func_map["data_kernel_weights_func"] = data_kernel_weights_func;
    func_map["data_kernels_func"] = data_kernels_func;
    func_map["w_data_func"] = w_data_re;
    func_map["w_reg_func"] = w_reg_re;
    func_map["xrp_func"] = xrp_re;
    func_map["next_xrp"] = next_xrp;
    return func_map;
}


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
        const Input &reg_kernels) {
    // A single iteration of conjugate gradient, takes X, R, P and updates them
    Func xrp_func("xrp_func");
    xrp_func(x, y, c, n) = xrp(x, y, c, n);
    Func reg_kernel_weights_func("reg_kernel_weights_func");
    reg_kernel_weights_func(n) = reg_kernel_weights(n);
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    RDom r_image(0, xrp.width(), 0, xrp.height(), 0, xrp.channels());
    RDom r_kernel(kernel);
    Func xrp_clamped = BoundaryConditions::repeat_edge(xrp_func,
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
 
    Func rTr("rTr");
    // alpha = r^T * r / p^T A^T A p
    rTr() = 0.f;
    rTr() += r(r_image.x, r_image.y, r_image.z) *
             r(r_image.x, r_image.y, r_image.z);
    // rTr() = print(rTr());
    Func Kp("Kp");
    Kp(x, y, c) = 0.f;
    Kp(x, y, c) += p(x + r_kernel.x - kernel.width()  / 2,
                     y + r_kernel.y - kernel.height() / 2,
                     c) *
                   kernel(r_kernel.x, r_kernel.y);
    Func KTKp("K^TKp");
    KTKp(x, y, c) = 0.f;
    KTKp(x, y, c) += Kp(x + r_kernel.x - kernel.width()  / 2,
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
    Func rKTrKp("rK^TrKp");
    rKTrKp(x, y, c, n) = 0.f;
    rKTrKp(x, y, c, n) += rKp(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                              y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                              c,
                              n) *
                          reg_kernels_func(reg_kernels.width()  - r_reg_kernel_xy.x - 1,
                                           reg_kernels.height() - r_reg_kernel_xy.y - 1,
                                           n);
    Func ATAp("A^TAp");
    ATAp(x, y, c) = KTKp(x, y, c);
    ATAp(x, y, c) += rKTrKp(x, y, c, r_reg_kernel_z.x) *
                     reg_kernel_weights_func(r_reg_kernel_z.x) *
                     reg_kernel_weights_func(r_reg_kernel_z.x);
    Func pTATAp("p^TA^TAp");
    pTATAp() = 0.f;
    pTATAp() += p(r_image.x, r_image.y, r_image.z) *
                ATAp(r_image.x, r_image.y, r_image.z);

    Func alpha("alpha");
    alpha() = rTr() / pTATAp();
    // x = x + alpha * p
    Func next_x("next_x");
    // next_x(x, y, c) = xk(x, y, c) + alpha() * p(x, y, c);
    next_x(x, y, c) = rKp(x, y, c, 0);
    // r = r - alpha * A^TAp
    Func next_r("next_r");
    //next_r(x, y, c) = r(x, y, c) - alpha() * ATAp(x, y, c);
    next_r(x, y, c) = r(x, y, c);
    // beta = nextR^TnextR / r^Tr
    Func nRTnR("nRTnR");
    nRTnR() = 0.f;
    nRTnR() += next_r(r_image.x, r_image.y, r_image.z) *
               next_r(r_image.x, r_image.y, r_image.z);
    Func beta("beta");
    beta() = nRTnR() / rTr();
    Func next_p("next_p");
    // next_p(x, y, c) = next_r(x, y, c) + beta() * p(x, y, c);
    next_p(x, y, c) = p(x, y, c);

    Func next_xrp("next_xrp");
    next_xrp(x, y, c, n) = 0.f;
    next_xrp(x, y, c, 0) = next_x(x, y, c);
    next_xrp(x, y, c, 1) = next_r(x, y, c);
    next_xrp(x, y, c, 2) = next_p(x, y, c);

    std::map<std::string, Func> func_map;
    func_map["reg_kernel_weights_func"] = reg_kernel_weights_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["xrp_func"] = xrp_func;
    func_map["xrp_clamped"] = xrp_clamped;
    func_map["xk"] = xk;
    func_map["r"] = r;
    func_map["p"] = p;
    func_map["rTr"] = rTr;
    func_map["Kp"] = Kp;
    func_map["KTKp"] = KTKp;
    func_map["rKp"] = rKp;
    func_map["rKTrKp"] = rKTrKp;
    func_map["ATAp"] = ATAp;
    func_map["pTATAp"] = pTATAp;
    func_map["alpha"] = alpha;
    func_map["next_x"] = next_x;
    func_map["next_r"] = next_r;
    func_map["nRTnR"] = nRTnR;
    func_map["beta"] = beta;
    func_map["next_p"] = next_p;
    func_map["next_xrp"] = next_xrp;
    return func_map;
}

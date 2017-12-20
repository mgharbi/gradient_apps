#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> deconv_cg_init(
        const Input &blurred,
        const Input &x0,
        const Input &kernel,
        const Input &reg_kernel_weights,
        const Input &reg_kernels,
        const Input &reg_target_kernels,
        const Input &w_kernel,
        const Input &w_reg_kernels) {
    // Initializing conjugate gradient
    // Want to solve A^TAx = A^Tb
    // A -> correlation with kernel
    // A^T -> convolution with kernel
    // Initializing r0 = A^Tb - A^TAx0
    Func reg_kernel_weights_func("reg_kernel_weights_func");
    reg_kernel_weights_func(n) = reg_kernel_weights(n);
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    Func reg_target_kernels_func("reg_target_kernel_func");
    reg_target_kernels_func(x, y, n) = reg_target_kernels(x, y, n);
    Func w_kernel_func("w_kernel_func");
    w_kernel_func(x, y, c) = w_kernel(x, y, c);
    Func w_reg_kernels_func("w_reg_kernels_func");
    w_reg_kernels_func(x, y, c, n) = w_reg_kernels(x, y, c, n);
    Func x0_func("x0_func");
    x0_func(x, y, c) = x0(x, y, c);
    RDom r_kernel(kernel);
    Func clamped_b = BoundaryConditions::repeat_edge(blurred);
    Func clamped_x0 = BoundaryConditions::repeat_edge(x0_func,
                {{Expr(0), Expr(x0.width())},
                 {Expr(0), Expr(x0.height())},
                 {Expr(), Expr()}});
    Func clamped_w_kernel = BoundaryConditions::repeat_edge(w_kernel_func,
                {{Expr(0), Expr(x0.width())},
                 {Expr(0), Expr(x0.height())},
                 {Expr(), Expr()}});
    Func clamped_w_reg_kernels = BoundaryConditions::repeat_edge(w_reg_kernels_func,
                {{Expr(0), Expr(x0.width())},
                 {Expr(0), Expr(x0.height())},
                 {Expr(), Expr()},
                 {Expr(), Expr()}});

    Func wkb("wkb");
    wkb(x, y, c) = clamped_w_kernel(x, y, c) * clamped_b(x, y, c);
    Func KTWb("K^TWb");
    KTWb(x, y, c) = 0.f;
    KTWb(x, y, c) += wkb(x + r_kernel.x - kernel.width()  / 2,
                         y + r_kernel.y - kernel.height() / 2,
                         c) *
                     kernel(kernel.width()  - r_kernel.x - 1,
                            kernel.height() - r_kernel.y - 1);
    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func wrkb("wrkb");
    wrkb(x, y, c, n) = clamped_w_reg_kernels(x, y, c, n) * clamped_b(x, y, c);
    Func rKTWb("rK^TWb");
    rKTWb(x, y, c, n) = 0.f;
    rKTWb(x, y, c, n) += wrkb(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                              y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                              c,
                              n) *
                         reg_target_kernels_func(r_reg_kernel_xy.x,
                                                 r_reg_kernel_xy.y,
                                                 n);
    Func ATWb("A^TWb");
    ATWb(x, y, c) = KTWb(x, y, c);
    ATWb(x, y, c) += rKTWb(x, y, c, r_reg_kernel_z.x) *
                     reg_kernel_weights_func(r_reg_kernel_z.x) *
                     reg_kernel_weights_func(r_reg_kernel_z.x);

    Func Kx0("Kx0");
    Kx0(x, y, c) = 0.f;
    Kx0(x, y, c) += clamped_x0(x + r_kernel.x - kernel.width()  / 2,
                               y + r_kernel.y - kernel.height() / 2,
                               c) *
                    kernel(r_kernel.x, r_kernel.y);
    Func WKx0("WKx0");
    WKx0(x, y, c) = Kx0(x, y, c) * clamped_w_kernel(x, y, c);
    Func KTWKx0("K^TWKx0");
    KTWKx0(x, y, c)  = 0.f;
    KTWKx0(x, y, c) += WKx0(x + r_kernel.x - kernel.width()  / 2,
                            y + r_kernel.y - kernel.height() / 2,
                            c) *
                        kernel(kernel.width()  - r_kernel.x - 1,
                               kernel.height() - r_kernel.y - 1);
    Func rKx0("rKx0");
    rKx0(x, y, c, n) = 0.f;
    rKx0(x, y, c, n) += clamped_x0(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                   y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                   c) *
                        reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func WrKx0("WrKx0");
    WrKx0(x, y, c, n) = rKx0(x, y, c, n) * clamped_w_reg_kernels(x, y, c, n);
    Func rKTWrKx0("rK^TWrKx0");
    rKTWrKx0(x, y, c, n) = 0.f;
    rKTWrKx0(x, y, c, n) += WrKx0(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                  y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                  c,
                                  n) *
                            reg_kernels_func(reg_kernels.width()  - r_reg_kernel_xy.x - 1,
                                             reg_kernels.height() - r_reg_kernel_xy.y - 1,
                                             n);

    Func ATWAx0("A^TWAx0");
    ATWAx0(x, y, c) = KTWKx0(x, y, c);
    ATWAx0(x, y, c) += rKTWrKx0(x, y, c, r_reg_kernel_z.x) *
                       reg_kernel_weights_func(r_reg_kernel_z.x) *
                       reg_kernel_weights_func(r_reg_kernel_z.x);

    Func r0("r0");
    r0(x, y, c) = ATWb(x, y, c) - ATWAx0(x, y, c);
    Func p0("p0");
    p0(x, y, c) = r0(x, y, c);
    Func xrp("xrp");
    xrp(x, y, c, n) = 0.f;
    xrp(x, y, c, 0) = x0_func(x, y, c);
    xrp(x, y, c, 1) = r0(x, y, c);
    xrp(x, y, c, 2) = p0(x, y, c);

    std::map<std::string, Func> func_map;

    func_map["x0_func"] = x0_func;
    func_map["reg_kernel_weights_func"] = reg_kernel_weights_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["reg_target_kernels_func"] = reg_target_kernels_func;
    func_map["w_kernel_func"] = w_kernel_func;
    func_map["w_reg_kernels_func"] = w_reg_kernels_func;
    func_map["xrp"] = xrp;
    return func_map;
}


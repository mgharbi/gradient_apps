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
        const Input &reg_target_kernels) {
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
    RDom r_kernel(kernel);
    Func clamped_b = BoundaryConditions::repeat_edge(blurred);
    Func clamped_x0 = BoundaryConditions::repeat_edge(x0);
    Func KTb("K^Tb");
    KTb(x, y, c) = 0.f;
    KTb(x, y, c) += clamped_b(x + r_kernel.x - kernel.width()  / 2,
                              y + r_kernel.y - kernel.height() / 2,
                              c) *
                    kernel(kernel.width()  - r_kernel.x - 1,
                           kernel.height() - r_kernel.y - 1);
    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func rKTb("rK^Tb");
    rKTb(x, y, c, n) = 0.f;
    rKTb(x, y, c, n) += clamped_b(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                  y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                  c) *
                        reg_target_kernels_func(r_reg_kernel_xy.x,
                                                r_reg_kernel_xy.y,
                                                n);
    Func ATb("A^Tb");
    ATb(x, y, c) = KTb(x, y, c);
    ATb(x, y, c) += rKTb(x, y, c, r_reg_kernel_z.x) *
                    reg_kernel_weights_func(r_reg_kernel_z.x) *
                    reg_kernel_weights_func(r_reg_kernel_z.x);

    Func Kx0("Kx0");
    Kx0(x, y, c)  = 0.f;
    Kx0(x, y, c) += clamped_x0(x + r_kernel.x - kernel.width()  / 2,
                               y + r_kernel.y - kernel.height() / 2,
                               c) *
                    kernel(r_kernel.x, r_kernel.y);
    Func KTKx0("K^TKx0");
    KTKx0(x, y, c)  = 0.f;
    KTKx0(x, y, c) += Kx0(x + r_kernel.x - kernel.width()  / 2,
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
    Func rKTrKx0("rK^TrKx0");
    rKTrKx0(x, y, c, n) = 0.f;
    rKTrKx0(x, y, c, n) += rKx0(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                c,
                                n) *
                           reg_kernels_func(reg_kernels.width()  - r_reg_kernel_xy.x - 1,
                                            reg_kernels.height() - r_reg_kernel_xy.y - 1,
                                            n);

    Func ATAx0("A^TAx0");
    ATAx0(x, y, c) = KTKx0(x, y, c);
    ATAx0(x, y, c) += rKTrKx0(x, y, c, r_reg_kernel_z.x) *
                      reg_kernel_weights_func(r_reg_kernel_z.x) *
                      reg_kernel_weights_func(r_reg_kernel_z.x);

    Func r0("r0");
    r0(x, y, c) = ATb(x, y, c) - ATAx0(x, y, c);
    Func p0("p0");
    p0(x, y, c) = r0(x, y, c);
    Func xrp("xrp");
    xrp(x, y, c, n) = 0.f;
    xrp(x, y, c, 0) = x0(x, y, c);
    xrp(x, y, c, 1) = r0(x, y, c);
    xrp(x, y, c, 2) = p0(x, y, c);

    std::map<std::string, Func> func_map;
    func_map["reg_kernel_weights_func"] = reg_kernel_weights_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["reg_target_kernels_func"] = reg_target_kernels_func;
    func_map["KTb"] = KTb;
    func_map["rKTb"] = rKTb;
    func_map["ATb"] = ATb;
    func_map["Kx0"] = Kx0;
    func_map["KTKx0"] = KTKx0;
    func_map["rKx0"] = rKx0;
    func_map["rKTrKx0"] = rKTrKx0;
    func_map["ATAx0"] = ATAx0;
    func_map["r0"] = r0;
    func_map["p0"] = p0;
    func_map["xrp"] = xrp;
    return func_map;
}


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
        const Input &data_kernel_weights,
        const Input &data_kernels,
        const Input &reg_kernel_weights,
        const Input &reg_kernels,
        const Input &reg_targets,
        const Input &precond_kernel,
        const Input &w_data,
        const Input &w_reg) {
    // Initializing conjugate gradient
    // Want to solve A^TAx = A^Tb
    // A -> correlation with kernel
    // A^T -> convolution with kernel
    // Initializing r0 = A^Tb - A^TAx0
    Func data_kernel_weights_func("data_kernel_weights_func");
    data_kernel_weights_func(n) = data_kernel_weights(n);
    Func data_kernels_func("data_kernels_func");
    data_kernels_func(x, y, n) = data_kernels(x, y, n);
    Func reg_kernel_weights_func("reg_kernel_weights_func");
    reg_kernel_weights_func(n) = reg_kernel_weights(n);
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    Func reg_targets_func("reg_targets_func");
    reg_targets_func(x, y, c, n) = reg_targets(x, y, c, n);
    Func precond_kernel_func("precond_kernel_func");
    precond_kernel_func(x, y) = precond_kernel(x, y);
    Func w_data_func("w_data_func");
    w_data_func(x, y, c, n) = w_data(x, y, c, n);
    Func w_reg_func("w_reg_func");
    w_reg_func(x, y, c, n) = w_reg(x, y, c, n);
    Func x0_func("x0_func");
    x0_func(x, y, c) = x0(x, y, c);

    RDom r_kernel(kernel);
    RDom r_data_kernel_xy(0, data_kernels.width(), 0, data_kernels.height());
    RDom r_data_kernel_z(0, data_kernels.channels());
    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func b_re = BoundaryConditions::repeat_edge(blurred);
    Func x0_re, clamped_x0;
    std::tie(x0_re, clamped_x0) = select_repeat_edge(x0_func, x0.width(), x0.height());
    Func rtarget_re, clamped_rtarget;
    std::tie(rtarget_re, clamped_rtarget) = select_repeat_edge(reg_targets_func, x0.width(), x0.height());
    Func w_data_re, clamped_w_data;
    std::tie(w_data_re, clamped_w_data) = select_repeat_edge(w_data_func, x0.width(), x0.height());
    Func w_reg_re, clamped_w_reg;
    std::tie(w_reg_re, clamped_w_reg) = select_repeat_edge(w_reg_func, x0.width(), x0.height());

    // Data term on right hand side
    Func wkb("wkb");
    wkb(x, y, c, n) = clamped_w_data(x, y, c, n) * b_re(x, y, c);
    Func dKTWb("dK^TWb");
    dKTWb(x, y, c, n) = 0.f;
    dKTWb(x, y, c, n) += wkb(x - r_data_kernel_xy.x + data_kernels.width()  / 2,
                             y - r_data_kernel_xy.y + data_kernels.height() / 2,
                             c,
                             n) *
                         data_kernels_func(r_data_kernel_xy.x, r_data_kernel_xy.y, n);
    Func wdKTWb("wdKTWb");
    wdKTWb(x, y, c) = 0.f;
    wdKTWb(x, y, c) += dKTWb(x, y, c, r_data_kernel_z) *
                       abs(data_kernel_weights_func(r_data_kernel_z));
    Func KTWb("K^TWb");
    KTWb(x, y, c) = 0.f;
    KTWb(x, y, c) += wdKTWb(x - r_kernel.x + kernel.width()  / 2,
                            y - r_kernel.y + kernel.height() / 2,
                            c) *
                     kernel(r_kernel.x, r_kernel.y);
   
    // Regularization term on right hand side
    Func wrkb("wrkb");
    wrkb(x, y, c, n) = clamped_w_reg(x, y, c, n) * clamped_rtarget(x, y, c, n);
    Func rKTWb("rK^TWb");
    rKTWb(x, y, c, n) = 0.f;
    rKTWb(x, y, c, n) += wrkb(x - r_reg_kernel_xy.x + reg_kernels.width()  / 2,
                              y - r_reg_kernel_xy.y + reg_kernels.height() / 2,
                              c,
                              n) *
                         reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func wrKTWb("wrK^TWb");
    wrKTWb(x, y, c) = 0.f;
    wrKTWb(x, y, c) += rKTWb(x, y, c, r_reg_kernel_z) *
                       abs(reg_kernel_weights_func(r_reg_kernel_z));

    // Right hand side
    Func ATWb("A^TWb");
    ATWb(x, y, c) = KTWb(x, y, c) + wrKTWb(x, y, c);

    // Data term at left hand side
    Func Kx0("Kx0");
    Kx0(x, y, c) = 0.f;
    Kx0(x, y, c) += clamped_x0(x + r_kernel.x - kernel.width()  / 2,
                               y + r_kernel.y - kernel.height() / 2,
                               c) *
                    kernel(r_kernel.x, r_kernel.y);
    Func dKx0("dKx0");
    dKx0(x, y, c, n) = 0.f;
    dKx0(x, y, c, n) += Kx0(x + r_data_kernel_xy.x - data_kernels.width()  / 2,
                            y + r_data_kernel_xy.y - data_kernels.height() / 2,
                            c) *
                        data_kernels_func(r_data_kernel_xy.x, r_data_kernel_xy.y, n);
    Func WdKx0("WdKx0");
    WdKx0(x, y, c, n) = dKx0(x, y, c, n) * clamped_w_data(x, y, c, n);
    Func dKTWdKx0("dK^TWdKx0");
    dKTWdKx0(x, y, c, n) = 0.f;
    dKTWdKx0(x, y, c, n) += WdKx0(x - r_data_kernel_xy.x + data_kernels.width()  / 2,
                                  y - r_data_kernel_xy.y + data_kernels.height() / 2,
                                  c, n) *
                            data_kernels_func(r_data_kernel_xy.x,
                                              r_data_kernel_xy.y,
                                              n);
    Func rdKTWdKx0("rdKTWdKx0");
    rdKTWdKx0(x, y, c) = 0.f;
    rdKTWdKx0(x, y, c) += dKTWdKx0(x, y, c, r_data_kernel_z) *
                          abs(data_kernel_weights_func(r_data_kernel_z));
    Func KTWKx0("K^TWKx0");
    KTWKx0(x, y, c)  = 0.f;
    KTWKx0(x, y, c) += rdKTWdKx0(x - r_kernel.x + kernel.width()  / 2,
                                 y - r_kernel.y + kernel.height() / 2,
                                 c) *
                       kernel(r_kernel.x, r_kernel.y);

    // Regularization term at left hand side
    Func rKx0("rKx0");
    rKx0(x, y, c, n) = 0.f;
    rKx0(x, y, c, n) += clamped_x0(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                   y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                   c) *
                        reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func WrKx0("WrKx0");
    WrKx0(x, y, c, n) = rKx0(x, y, c, n) * clamped_w_reg(x, y, c, n);
    Func rKTWrKx0("rK^TWrKx0");
    rKTWrKx0(x, y, c, n) = 0.f;
    rKTWrKx0(x, y, c, n) += WrKx0(x - r_reg_kernel_xy.x + reg_kernels.width()  / 2,
                                  y - r_reg_kernel_xy.y + reg_kernels.height() / 2,
                                  c,
                                  n) *
                            reg_kernels_func(r_reg_kernel_xy.x,
                                             r_reg_kernel_xy.y,
                                             n);
    Func wrKTWrKx0("wrKTWrKx0");
    wrKTWrKx0(x, y, c) = 0.f;
    wrKTWrKx0(x, y, c) += rKTWrKx0(x, y, c, r_reg_kernel_z) *
                          abs(reg_kernel_weights_func(r_reg_kernel_z));

    // Left hand side
    Func ATWAx0("A^TWAx0");
    ATWAx0(x, y, c) = KTWKx0(x, y, c) + wrKTWrKx0(x, y, c);

    Func r0("r0");
    r0(x, y, c) = ATWb(x, y, c) - ATWAx0(x, y, c);
    RDom r_precond_kernel(precond_kernel);
    Func Pr0("Pr0");
    Pr0(x, y, c) = 0.f;
    Pr0(x, y, c) += r0(x + r_precond_kernel.x - precond_kernel.width() / 2,
                       y + r_precond_kernel.y - precond_kernel.height() / 2,
			           c) *
	                precond_kernel_func(r_precond_kernel.x, r_precond_kernel.y);
    Func z0("z0");
    z0(x, y, c) = 0.f;
    z0(x, y, c) += Pr0(x - r_precond_kernel.x + precond_kernel.width() / 2,
                       y - r_precond_kernel.y + precond_kernel.height() / 2,
        	  	       c) *
  	               precond_kernel_func(r_precond_kernel.x,
                                       r_precond_kernel.y);

    Func p0("p0");
    p0(x, y, c) = z0(x, y, c);
    Func xrp("xrp");
    xrp(x, y, c, n) = 0.f;
    xrp(x, y, c, 0) = x0_re(x, y, c);
    xrp(x, y, c, 1) = r0(x, y, c);
    xrp(x, y, c, 2) = p0(x, y, c);
    xrp(x, y, c, 3) = z0(x, y, c);

    std::map<std::string, Func> func_map;
    func_map["x0_func"] = x0_re;
    func_map["data_kernel_weights_func"] = data_kernel_weights_func;
    func_map["data_kernels_func"] = data_kernels_func;
    func_map["reg_kernel_weights_func"] = reg_kernel_weights_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["reg_targets_func"] = reg_targets_func;
    func_map["precond_kernel_func"] = precond_kernel_func;
    func_map["w_data_func"] = w_data_re;
    func_map["w_reg_func"] = w_reg_re;
    func_map["xrp"] = xrp;
    return func_map;
}


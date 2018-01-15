#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
Func deconv_cg_init(
        const Input &blurred,
        const Input &x0,
        const Input &kernel,
        const Input &data_kernel_weights,
        const Input &data_kernels,
        const Input &reg_kernel_weights,
        const Input &reg_kernels,
        const Input &reg_targets,
        const Input &w_data,
        const Input &w_reg) {
    // Initializing conjugate gradient
    // Want to solve A^TAx = A^Tb
    // A -> correlation with kernel
    // A^T -> convolution with kernel
    // Initializing r0 = A^Tb - A^TAx0
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
    Func clamped_b = BoundaryConditions::repeat_edge(blurred);
    Func clamped_x0 = BoundaryConditions::repeat_edge(x0);
    Func clamped_reg_targets = BoundaryConditions::repeat_edge(reg_targets);
    Func clamped_w_data = BoundaryConditions::repeat_edge(w_data);
    Func clamped_w_reg = BoundaryConditions::repeat_edge(w_reg);

    // Note that on right hand side all kernels need to be transposed
    // Data term on right hand side
    Func wkb("wkb");
    wkb(x, y, c, n) = clamped_w_data(x, y, c, n) * clamped_b(x, y, c);
    Func dKTWb("dK^TWb");
    dKTWb(x, y, c, n) = sum(wkb(x - r_dk.x + dkw / 2, y - r_dk.y + dkh / 2, c, n) *
                            data_kernels(r_dk.x, r_dk.y, n));
    Func wdKTWb("wdKTWb");
    wdKTWb(x, y, c) = sum(dKTWb(x, y, c, r_dk_c) *
                          abs(data_kernel_weights(r_dk_c)));
    Func KTWb("K^TWb");
    KTWb(x, y, c) = sum(wdKTWb(x - r_k.x + kw / 2, y - r_k.y + kh / 2, c) *
                        kernel(r_k.x, r_k.y));
    // Regularization term on right hand side
    Func wrkb("wrkb");
    wrkb(x, y, c, n) = clamped_w_reg(x, y, c, n) * clamped_reg_targets(x, y, c, n);
    Func rKTWb("rK^TWb");
    rKTWb(x, y, c, n) = sum(wrkb(x - r_rk.x + rkw / 2, y - r_rk.y + rkh / 2, c, n) *
                            reg_kernels(r_rk.x, r_rk.y, n));
    Func wrKTWb("wrK^TWb");
    wrKTWb(x, y, c) = sum(rKTWb(x, y, c, r_rk_c) *
                          abs(reg_kernel_weights(r_rk_c)));
    // Right hand side
    Func ATWb("A^TWb");
    ATWb(x, y, c) = KTWb(x, y, c) + wrKTWb(x, y, c);

    // Note that on left hand side we need to apply the kernels twice:
    // once original and once transpose version
    // Data term at left hand side
    Func Kx0("Kx0");
    Kx0(x, y, c) = sum(clamped_x0(x + r_k.x - kw / 2, y + r_k.y - kh / 2, c) *
                       kernel(r_k.x, r_k.y));
    Func dKx0("dKx0");
    dKx0(x, y, c, n) = sum(Kx0(x + r_dk.x - dkw / 2, y + r_dk.y - dkh / 2, c) *
                           data_kernels(r_dk.x, r_dk.y, n));
    Func WdKx0("WdKx0");
    WdKx0(x, y, c, n) = dKx0(x, y, c, n) * clamped_w_data(x, y, c, n);
    Func dKTWdKx0("dK^TWdKx0");
    dKTWdKx0(x, y, c, n) = sum(WdKx0(x - r_dk.x + dkw / 2, y - r_dk.y + dkh / 2, c, n) *
                               data_kernels(r_dk.x, r_dk.y, n));
    Func rdKTWdKx0("rdKTWdKx0");
    rdKTWdKx0(x, y, c) = sum(dKTWdKx0(x, y, c, r_dk_c) *
                             abs(data_kernel_weights(r_dk_c)));
    Func KTWKx0("K^TWKx0");
    KTWKx0(x, y, c) = sum(rdKTWdKx0(x - r_k.x + kw / 2, y - r_k.y + kh / 2, c) *
                          kernel(r_k.x, r_k.y));

    // Regularization term at left hand side
    Func rKx0("rKx0");
    rKx0(x, y, c, n) = sum(clamped_x0(x + r_rk.x - rkw / 2, y + r_rk.y - rkh / 2, c) *
                           reg_kernels(r_rk.x, r_rk.y, n));
    Func WrKx0("WrKx0");
    WrKx0(x, y, c, n) = rKx0(x, y, c, n) * clamped_w_reg(x, y, c, n);
    Func rKTWrKx0("rK^TWrKx0");
    rKTWrKx0(x, y, c, n) = sum(WrKx0(x - r_rk.x + rkw / 2, y - r_rk.y + rkh / 2, c, n) *
                               reg_kernels(r_rk.x, r_rk.y, n));
    Func wrKTWrKx0("wrKTWrKx0");
    wrKTWrKx0(x, y, c) = sum(rKTWrKx0(x, y, c, r_rk_c) *
                             abs(reg_kernel_weights(r_rk_c)));

    // Left hand side
    Func ATWAx0("A^TWAx0");
    ATWAx0(x, y, c) = KTWKx0(x, y, c) + wrKTWrKx0(x, y, c);

    Func r0("r0");
    r0(x, y, c) = ATWb(x, y, c) - ATWAx0(x, y, c);

    Func p0("p0");
    p0(x, y, c) = r0(x, y, c);
    Func xrp("xrp");
    xrp(x, y, c, n) = 0.f;
    xrp(x, y, c, 0) = x0(x, y, c);
    xrp(x, y, c, 1) = r0(x, y, c);
    xrp(x, y, c, 2) = p0(x, y, c);

    return xrp;
}


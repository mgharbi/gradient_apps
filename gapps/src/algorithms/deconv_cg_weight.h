#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> deconv_cg_weight(
        const Input &blurred,
        const Input &current,
        const Input &reg_kernels,
        const Input &reg_target_kernels,
        const Input &reg_powers) {
    // Compute the residuals with the power of p-2
    Func current_func("current_func");
    current_func(x, y, c) = current(x, y, c);
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    Func reg_target_kernels_func("reg_target_kernel_func");
    reg_target_kernels_func(x, y, n) = reg_target_kernels(x, y, n);
    Func reg_powers_func("reg_powers_func");
    reg_powers_func(n) = reg_powers(n);

    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func clamped_blurred = BoundaryConditions::repeat_edge(blurred);
    Func clamped_current = BoundaryConditions::repeat_edge(current_func,
                {{Expr(0), Expr(current.width())},
                 {Expr(0), Expr(current.height())},
                 {Expr(), Expr()}});

    Func rKc("rKc");
    rKc(x, y, c, n) = 0.f;
    rKc(x, y, c, n) += clamped_current(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                       y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                       c) *
                       reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);

    Func rtKb("rtKb");
    rtKb(x, y, c, n) = 0.f;
    rtKb(x, y, c, n) += clamped_blurred(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                        y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                        c) *
                        reg_target_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);

    Func weights("weights");
    weights(x, y, c, n) =
        1.f / (max(1e-4f, pow(max(abs(rKc(x, y, c, n) - rtKb(x, y, c, n)), 1e-4f),
                reg_powers_func(n) - 2.f)));

    std::map<std::string, Func> func_map;
    func_map["current_func"] = current_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["reg_target_kernels_func"] = reg_target_kernels_func;
    func_map["reg_powers_func"] = reg_powers_func;
    func_map["weights"] = weights;
    return func_map;
}

 

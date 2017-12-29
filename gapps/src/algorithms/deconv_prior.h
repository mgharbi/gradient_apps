#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> deconv_prior(
        const Input &f,
        const Input &reg_kernels,
        const Input &threshold) {
    // Compute the convolution of reg_kernels with f
    // and compute the prior weights
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    Func f_func("f_func");
    f_func(x, y, c) = f(x, y, c);
    Func threshold_func("threshold_func");
    threshold_func() = threshold;

    Func clamped_f = BoundaryConditions::repeat_edge(f_func,
                {{Expr(0), Expr(f.width())},
                 {Expr(0), Expr(f.height())},
                 {Expr(), Expr()}});

    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func rKf("rKf");
    rKf(x, y, c, n) = 0.f;
    rKf(x, y, c, n) += clamped_x0(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                  y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                  c) *
                       reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func w("w");
    w(x, y, c, n) = rKf(x, y, c, n) /
            (pow(threshold / (rKf(x, y, c, n) + 1e-3f), 4.f) + 1.f);

    std::map<std::string, Func> func_map;
    func_map["f_func"] = f_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["threshold_func"] = threshold_func;
    func_map["w"] = w;
    return func_map;
}

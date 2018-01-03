#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n"), j("j");

template <typename Input>
std::map<std::string, Func> deconv_cg_weight(
        const Input &blurred,
        const Input &current,
        const Input &reg_kernels,
        const Input &reg_targets,
        const Input &gmm_weights,
        const Input &gmm_invvars) {
    // Compute the residuals with the power of p-2
    Func current_func("current_func");
    current_func(x, y, c) = current(x, y, c);
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    Func reg_targets_func("reg_targets_func");
    reg_targets_func(x, y, c, n) = reg_targets(x, y, c, n);
    Func gmm_weights_func("gmm_weights_func");
    gmm_weights_func(n, j) = gmm_weights(n, j);
    Func gmm_invvars_func("gmm_invvars_func");
    gmm_invvars_func(n, j) = gmm_invvars(n, j);

    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func clamped_blurred = BoundaryConditions::repeat_edge(blurred);
    Func current_re, clamped_current;
    std::tie(current_re, clamped_current) = select_repeat_edge(current_func, current.width(), current.height());
    Func rtarget_re, clamped_rtarget;
    std::tie(rtarget_re, clamped_rtarget) = select_repeat_edge(reg_targets_func, reg_targets.width(), reg_targets.height());

    Func rKc("rKc");
    rKc(x, y, c, n) = 0.f;
    rKc(x, y, c, n) += clamped_current(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                                       y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                                       c) *
                       reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func dist("dist");
    dist(x, y, c, n) = (rKc(x, y, c, n) - clamped_rtarget(x, y, c, n)) * 
                       (rKc(x, y, c, n) - clamped_rtarget(x, y, c, n));

    Func gmm_likelihood("gmm_likelihood");
    gmm_likelihood(x, y, c, n, j) = gmm_weights_func(n, j) * sqrt(gmm_invvars_func(n, j)) *
        exp(-0.5f * dist(x, y, c, n) * gmm_invvars_func(n, j));

    RDom r_gmm(0, gmm_weights.height());
    Func weights("weights");
    weights(x, y, c, n) =
        sum(gmm_invvars_func(n, r_gmm) * gmm_likelihood(x, y, c, n, r_gmm)) /
        sum(gmm_likelihood(x, y, c, n, r_gmm));

    std::map<std::string, Func> func_map;
    func_map["current_func"] = current_re;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["reg_targets_func"] = rtarget_re;
    func_map["gmm_weights_func"] = gmm_weights_func;
    func_map["gmm_invvars_func"] = gmm_invvars_func;
    func_map["weights"] = weights;
    return func_map;
}


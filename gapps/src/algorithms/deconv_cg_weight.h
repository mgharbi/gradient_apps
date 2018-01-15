#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n"), j("j");

Expr square(Expr e) {
    return e * e;
}

template <typename Input>
Func deconv_cg_weight(
        const Input &blurred,
        const Input &current,
        const Input &reg_kernels,
        const Input &reg_targets,
        const Input &reg_powers) {
    // Compute IRLS weight for p norm
    Expr rkw = reg_kernels.width();
    Expr rkh = reg_kernels.height();
    RDom r_rk(0, rkw, 0, rkh);
    RDom r_rk_c(0, reg_kernels.channels());
    Func clamped_blurred = BoundaryConditions::repeat_edge(blurred);
    Func clamped_current = BoundaryConditions::repeat_edge(current);
    Func clamped_reg_targets = BoundaryConditions::repeat_edge(reg_targets);

    Func rKc("rKc");
    rKc(x, y, c, n) = sum(clamped_current(x + r_rk.x - rkw / 2, y + r_rk.y - rkh / 2, c) *
                          reg_kernels(r_rk.x, r_rk.y, n));
    Func weights("weights");
    weights(x, y, c, n) = 1.f /
        max(1e-4f, pow(abs(rKc(x, y, c, n) - reg_targets(x, y, c, n)), reg_powers(n)));
    return weights;
}


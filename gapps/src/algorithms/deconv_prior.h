#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
Func deconv_prior(const Input &f,
                  const Input &reg_kernels,
                  const Input &thresholds) {
    // Compute the convolution of reg_kernels with f
    // and compute the prior weights
    Func f_clamped = BoundaryConditions::repeat_edge(f);
    Expr rkw = reg_kernels.width();
    Expr rkh = reg_kernels.height();
    RDom r_rk(0, rkw, 0, rkh);
    RDom r_rk_c(0, reg_kernels.channels());
    Func rKf("rKf");
    rKf(x, y, c, n) = sum(f_clamped(x + r_rk.x - rkw / 2, y + r_rk.y - rkh / 2, c) *
                          reg_kernels(r_rk.x, r_rk.y, n));
    Func weights("weights");
    Expr f2 = rKf(x, y, c, n) * rKf(x, y, c, n);
    Expr f4 = f2 * f2;
    Expr t2 = thresholds(n) * thresholds(n);
    Expr t4 = t2 * t2;
    weights(x, y, c, n) = f4 * rKf(x, y, c, n) / (t4 + f4 + 1e-6f);

    return weights;
}


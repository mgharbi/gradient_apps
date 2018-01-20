#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
Func bilinear_resampling(const Input &input,
                         const Input &warp) {
    Func clamped = Halide::BoundaryConditions::constant_exterior(input);

    Expr width = input.dim(0).extent();
    Expr height = input.dim(1).extent();

    Expr dx = warp(x, y, 0, n);
    Expr dy = warp(x, y, 1, n);

    // Convert back to image space
    Expr new_x = clamp(x+dx, -1.0f, cast<float>(width));
    Expr new_y = clamp(y+dy, -1.0f, cast<float>(height));

    // Bilinear interpolation
    Expr fx = cast<int>(floor(new_x));
    Expr fy = cast<int>(floor(new_y));
    Expr wx = new_x - fx;
    Expr wy = new_y - fy;

    Func output("output");
    output(x, y, c, n) =
        clamped(fx,   fy,   c, n)*(1.0f-wx)*(1.0f-wy)
      + clamped(fx,   fy+1, c, n)*(1.0f-wx)*(     wy)
      + clamped(fx+1, fy,   c, n)*(     wx)*(1.0f-wy)
      + clamped(fx+1, fy+1, c, n)*(     wx)*(     wy);

    return output;
}

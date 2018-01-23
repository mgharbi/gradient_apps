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
    // Func clamped = Halide::BoundaryConditions::constant_exterior(input);

    Expr width = input.dim(0).extent();
    Expr height = input.dim(1).extent();

    // Match pytorch's coordinates, which are in [-1, 1]
    Expr dx = 0.5f*(warp(x, y, 0, n) + 1.0f);
    Expr dy = 0.5f*(warp(x, y, 1, n) + 1.0f);

    // Convert back to image space
    Expr new_x = clamp(dx*width, 0.0f, cast<float>(width));
    Expr new_y = clamp(dy*height, 0.0f, cast<float>(height));

    // Bilinear interpolation
    Expr fx = clamp(cast<int>(floor(new_x)), 0, width-2);
    Expr fy = clamp(cast<int>(floor(new_y)), 0, height-2);
    Expr wx = new_x - fx;
    Expr wy = new_y - fy;

    Func output("output");
    output(x, y, c, n) =
        input(fx,   fy,   c, n)*(1.0f-wx)*(1.0f-wy)
      + input(fx,   fy+1, c, n)*(1.0f-wx)*(     wy)
      + input(fx+1, fy,   c, n)*(     wx)*(1.0f-wy)
      + input(fx+1, fy+1, c, n)*(     wx)*(     wy);

    return output;
}

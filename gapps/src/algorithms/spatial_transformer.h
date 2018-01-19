#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
Func spatial_transformer(const Input &input,
                         const Input &affine_mtx) {
    Expr width = input.dim(0).extent();
    Expr height = input.dim(1).extent();

    // Normalize image coordinates to [-1, 1]^2
    Expr nrm_x = 2.0f*(x * 1.0f / width) - 1.0f;
    Expr nrm_y = 2.0f*(y * 1.0f / height) - 1.0f;

    // Normalized sampling location
    Expr xformed_x = 
          affine_mtx(0, 0, n)*nrm_x 
        + affine_mtx(1, 0, n)*nrm_y 
        + affine_mtx(2, 0, n);
    Expr xformed_y = 
          affine_mtx(0, 1, n)*nrm_x 
        + affine_mtx(1, 1, n)*nrm_y 
        + affine_mtx(2, 1, n);

    // Convert back to image space
    Expr new_x = clamp(
        width*0.5f*(xformed_x + 1.0f),
        0.0f, cast<float>(width-1));
    Expr new_y = clamp(
        height*0.5f*(xformed_y + 1.0f),
        0.0f, cast<float>(height-1));

    // Bilinear interpolation
    Expr fx = clamp(cast<int>(floor(new_x)), 0, width-2);
    Expr fy = clamp(cast<int>(floor(new_y)), 0, height-2);
    Expr wx = (new_x - fx);
    Expr wy = (new_y - fy);

    Func output("output");
    output(x, y, c, n) = 
        input(fx,   fy,   c, n)*(1.0f-wx)*(1.0f-wy)
      + input(fx,   fy+1, c, n)*(1.0f-wx)*(     wy)
      + input(fx+1, fy,   c, n)*(     wx)*(1.0f-wy)
      + input(fx+1, fy+1, c, n)*(     wx)*(     wy);

    return output;
}

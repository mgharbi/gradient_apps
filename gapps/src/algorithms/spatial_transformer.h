#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), z("z"), c("c"), n("n");

template <typename Input>
Func spatial_transformer(const Input &input,
                         const Input &affine_mtx) {
    Expr width = input.dim(0).extent();
    Expr height = input.dim(1).extent();

    // Normalize image coordinates to [-1, 1]^2
    Expr nrm_x = 2.0f*(x * 1.0f / width) - 1.0f;
    Expr nrm_y = 2.0f*(y * 1.0f / height) - 1.0f;
    Func nrm_coords;
    nrm_coords(x, y, z) = undef<float>();
    nrm_coords(x, y, 0) = 2.f * (x * 1.f / width) - 1.f;
    nrm_coords(x, y, 1) = 2.f * (y * 1.f / height) - 1.f;
    nrm_coords(x, y, 2) = 1.f;

    // Convert back to image space
    RDom rm(0, 3);
    Func xformed("xformed");
    xformed(x, y, z, n) += affine_mtx(rm, z, n) * nrm_coords(x, y, rm);
    xformed(x, y, z, n) = 0.5f * (xformed(x, y, z, n) + 1.f);

    // Bilinear interpolation
    Expr new_x = xformed(x, y, 0, n), new_y = xformed(x, y, 1, n);
    Expr fx = clamp(cast<int>(floor(width * new_x)), 0, width-2);
    Expr fy = clamp(cast<int>(floor(height * new_y)), 0, height-2);
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

#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> bilinear_resampling(
        const Input &input,
        const Input &warp
        ) {
    Func f_input("f_input");
    f_input(x, y, c, n) = Halide::BoundaryConditions::constant_exterior(
        input, 0)(x, y, c, n);
    Func f_warp("f_warp");
    f_warp(x, y, c, n) = warp(x, y, c, n);

    Expr width = input.dim(0).extent();
    Expr height = input.dim(1).extent();

    Expr dx = f_warp(x, y, 0, n);
    Expr dy = f_warp(x, y, 1, n);

    // Convert back to image space
    Expr new_x = clamp(x+dx,
        -1.0f, cast<float>(width));
    Expr new_y = clamp(y+dy,
        -1.0f, cast<float>(height));

    // Bilinear interpolation
    Expr fx = cast<int>(floor(new_x));
    Expr fy = cast<int>(floor(new_y));
    Expr wx = (new_x - fx);
    Expr wy = (new_y - fy);

    Func f_output("f_output");
    f_output(x, y, c, n) =
        f_input(fx,   fy,   c, n)*(1.0f-wx)*(1.0f-wy)
      + f_input(fx,   fy+1, c, n)*(1.0f-wx)*(     wy)
      + f_input(fx+1, fy,   c, n)*(     wx)*(1.0f-wy)
      + f_input(fx+1, fy+1, c, n)*(     wx)*(     wy);

    std::map<std::string, Func> func_map;
    func_map["input"]  = input;
    func_map["warp"]  = f_warp;
    func_map["output"]  = f_output;

    return func_map;
}

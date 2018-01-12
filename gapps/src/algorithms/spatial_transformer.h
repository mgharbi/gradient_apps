#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> spatial_transformer(
        const Input &input,
        const Input &affine_mtx
        ) {
    Func f_input("f_input");
    f_input(x, y, c, n) = Halide::BoundaryConditions::constant_exterior(
        input, 0)(x, y, c, n);
    Func f_affine_mtx("f_affine_mtx");
    f_affine_mtx(x, y, n) = affine_mtx(x, y, n);

    // Normalize image coordinates to [-1, 1]^2
    Expr nrm_x = 2.0f*(x * 1.0f / input.dim(0).extent()) - 1.0f;
    Expr nrm_y = 2.0f*(y * 1.0f / input.dim(1).extent()) - 1.0f;

    // Sampling location
    Expr new_x = f_affine_mtx(0, 0, n)*nrm_x 
               + f_affine_mtx(0, 1, n)*nrm_y 
               + f_affine_mtx(0, 2, n);

    Expr new_y = f_affine_mtx(1, 0, n)*nrm_x 
               + f_affine_mtx(1, 1, n)*nrm_y 
               + f_affine_mtx(1, 2, n);

    // Bilinear interpolation
    Expr fx = cast<int>(floor(new_x));
    Expr fy = cast<int>(floor(new_y));
    Expr wx = (new_x - fx);
    Expr wy = (new_y - fy);

    Func f_output("f_output");
    f_output(x, y, c, n) = 
        f_input(fx,   fy,   c, n)*(1-wx)*(1-wy)
      + f_input(fx,   fy+1, c, n)*(1-wx)*(  wy)
      + f_input(fx+1, fy,   c, n)*(  wx)*(1-wy)
      + f_input(fx+1, fy+1, c, n)*(  wx)*(  wy);

    std::map<std::string, Func> func_map;
    func_map["input"]  = input;
    func_map["affine_mtx"]  = f_affine_mtx;
    func_map["output"]  = f_output;

    return func_map;
}

#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), ci("ci"), co("co"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> bilateral_slice_apply(
        const Input &grid,
        const Input &guide,
        const Input &input) {

    Func f_grid("f_grid");
    f_grid(x, y, z, c, n) =
      Halide::BoundaryConditions::repeat_edge(grid)(x, y, z, c, n);
    Func f_guide("f_guide");
    f_guide(x, y, n) =
      Halide::BoundaryConditions::repeat_edge(guide)(x, y, n);
    Func f_input("f_input");
    f_input(x, y, ci, n) =
      Halide::BoundaryConditions::repeat_edge(input)(x, y, ci, n);

    Expr gw = grid.dim(0).extent();
    Expr gh = grid.dim(1).extent();
    Expr gd = grid.dim(2).extent();
    Expr w = input.dim(0).extent();
    Expr h = input.dim(1).extent();
    Expr nci = input.dim(2).extent();

    // Enclosing voxel
    // Limit the maximum reduction size to generate
    // better Halide code
    constexpr float eps = 1e-3f;
    Expr gx = (x+0.5f)*max(gw/(1.0f*w), eps);
    Expr gy = (y+0.5f)*max(gh/(1.0f*h), eps);
    Expr gz = clamp(f_guide(x, y, n), 0.0f, 1.0f)*gd;

    Expr fx = cast<int>(floor(gx-0.5f));
    Expr fy = cast<int>(floor(gy-0.5f));
    Expr fz = cast<int>(floor(gz-0.5f));
    Expr cx = fx+1;
    Expr cy = fy+1;
    Expr cz = fz+1;

    Expr wx = abs(gx-0.5f - fx);
    Expr wy = abs(gy-0.5f - fy);
    Expr wz = abs(gz-0.5f - fz);

    // Slice affine coeffs
    Func affine_coeffs("affine_coeffs");
    affine_coeffs(x, y, c, n) =
         f_grid(fx, fy, fz, c, n)*(1.f - wx)*(1.f - wy)*(1.f - wz)
       + f_grid(fx, fy, cz, c, n)*(1.f - wx)*(1.f - wy)*(      wz)
       + f_grid(fx, cy, fz, c, n)*(1.f - wx)*(      wy)*(1.f - wz)
       + f_grid(fx, cy, cz, c, n)*(1.f - wx)*(      wy)*(      wz)
       + f_grid(cx, fy, fz, c, n)*(      wx)*(1.f - wy)*(1.f - wz)
       + f_grid(cx, fy, cz, c, n)*(      wx)*(1.f - wy)*(      wz)
       + f_grid(cx, cy, fz, c, n)*(      wx)*(      wy)*(1.f - wz)
       + f_grid(cx, cy, cz, c, n)*(      wx)*(      wy)*(      wz);

    // Apply them to the input
    Func output("output");
    RDom r(0, nci);
    output(x, y, co, n) = affine_coeffs(x, y, co*(nci+1) + nci, n);
    output(x, y, co, n) += 
      affine_coeffs(x, y, co*(nci+1) + r, n)*f_input(x, y, r, n);

    std::map<std::string, Func> func_map;
    func_map["f_grid"]  = f_grid;
    func_map["f_guide"]  = f_guide;
    func_map["f_input"]  = f_input;
    func_map["output"] = output;
    return func_map;
}


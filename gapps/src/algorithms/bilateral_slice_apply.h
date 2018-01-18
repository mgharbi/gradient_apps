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

    Expr gw = guide.dim(0).extent();
    Expr gh = guide.dim(1).extent();
    Expr gd = guide.dim(2).extent();
    Expr w = input.dim(0).extent();
    Expr h = input.dim(1).extent();
    Expr nci = input.dim(2).extent();

    // Enclosing voxel
    Expr gx = (x+0.5f)*gw/(1.0f*w);
    Expr gy = (y+0.5f)*gh/(1.0f*h);
    Expr gz = clamp(f_guide(x, y, n), 0.0f, 1.0f)*gd;

    Expr fx = max(cast<int>(floor(gx-0.5f)), 0);
    Expr fy = max(cast<int>(floor(gy-0.5f)), 0);
    Expr fz = max(cast<int>(floor(gz-0.5f)), 0);
    Expr cx = min(fx+1, gw-1);
    Expr cy = min(fy+1, gh-1);
    Expr cz = min(fz+1, gd-1);

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


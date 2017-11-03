#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), n("n"), ci("ci"), co("co");

template <typename Input>
std::map<std::string, Func> bilateral_layer(
        const Input &input,
        const Input &guide,
        const Input &filter,
        const Expr &sigma_x,
        const Expr &sigma_y,
        const Expr sigma_z) {
    Func f_input("f_input");
    f_input(x, y, ci, n) = Halide::BoundaryConditions::repeat_edge(input)(x, y, ci, n);
    Func f_guide("f_guide");
    f_guide(x, y, n) = Halide::BoundaryConditions::repeat_edge(guide)(x, y, n);

    // Splat in z
    Expr guide_pos = clamp(f_guide(x, y, n)*sigma_z, 0, cast<float>(sigma_z));
    Expr lower_bin = cast<int>(floor(guide_pos));
    Expr upper_bin = cast<int>(ceil(guide_pos));
    Expr w = guide_pos - lower_bin;
    Func f_splatz("splat_z");
    f_splatz(x, y, z, ci, n) = 0.0f;
    f_splatz(x, y, lower_bin, ci, n) += (1-w)*f_input(x, y, ci, n);
    f_splatz(x, y, upper_bin, ci, n) += w*f_input(x, y, ci, n);

    // Downsample grid
    Expr normalization = 1.0f / (cast<float>(sigma_x) * cast<float>(sigma_y));
    RDom rgrid(0, sigma_x, 0, sigma_y);
    Func f_grid("bilateral_grid");
    f_grid(x, y, z, ci, n) = 0.f;
    f_grid(x, y, z, ci, n) += 
      normalization*f_splatz(x*sigma_x + rgrid.x, y*sigma_y + rgrid.y,
                             clamp(z, 0, sigma_z) , ci, n);

    // Perform 3D filtering in the grid
    Expr kw = filter.dim(0).extent();
    Expr kh = filter.dim(1).extent();
    Expr kd = filter.dim(2).extent();
    Expr in_chans = filter.dim(3).extent();
    RDom r(0, kw, 0, kh, 0, kd, 0, in_chans);
    Func f_filter("f_filter");
    f_filter(x, y, z, ci, co) = filter(x, y, z, ci, co);
    Func f_conv("conv");
    f_conv(x, y, z, co, n)  = 0.f;
    // TODO: center kernel
    f_conv(x, y, z, co, n) += f_filter(r[0], r[1], r[2], r[3], co) *
                              f_grid(x + r[0], y + r[1], z + r[2], r[3], n);

    // Enclosing voxel
    Expr gx = (x+0.5f)/(1.0f*sigma_x);
    Expr gy = (y+0.5f)/(1.0f*sigma_y);
    Expr gz = guide_pos;
    Expr fx = cast<int>(floor(gx-0.5f));
    Expr fy = cast<int>(floor(gy-0.5f));
    Expr fz = cast<int>(floor(gz));
    Expr cx = fx+1;
    Expr cy = fy+1;
    Expr cz = cast<int>(ceil(gz));
    Expr wx = gx-0.5f - fx;
    Expr wy = gy-0.5f - fy;
    Expr wz = gz - fz;

    // trilerp
    Func f_output("f_output");
    f_output(x, y, co, n) =
         f_conv(fx, fy, fz, co, n)*(1.f - wx)*(1.f - wy)*(1.f - wz)
       + f_conv(fx, fy, cz, co, n)*(1.f - wx)*(1.f - wy)*(      wz)
       + f_conv(fx, cy, fz, co, n)*(1.f - wx)*(      wy)*(1.f - wz)
       + f_conv(fx, cy, cz, co, n)*(1.f - wx)*(      wy)*(      wz)
       + f_conv(cx, fy, fz, co, n)*(      wx)*(1.f - wy)*(1.f - wz)
       + f_conv(cx, fy, cz, co, n)*(      wx)*(1.f - wy)*(      wz)
       + f_conv(cx, cy, fz, co, n)*(      wx)*(      wy)*(1.f - wz)
       + f_conv(cx, cy, cz, co, n)*(      wx)*(      wy)*(      wz);

    std::map<std::string, Func> func_map;
    func_map["input"]  = f_input;
    func_map["guide"]  = f_guide;
    func_map["grid"]   = f_grid;
    func_map["filter"] = f_filter;
    func_map["conv"]   = f_conv;
    func_map["output"] = f_output;

    return func_map;
}

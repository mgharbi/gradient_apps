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
        const Input &filter) {
    Func f_input("f_input");
    f_input(x, y, ci, n) = 
      Halide::BoundaryConditions::repeat_edge(input)(x, y, ci, n);
    Func f_guide("f_guide");
    f_guide(x, y, n) = Halide::BoundaryConditions::repeat_edge(guide)(x, y, n);
    Func f_filter("f_filter");
    f_filter(x, y, z, ci, co) = filter(x, y, z, ci, co);

    int sigma_s = 8;
    int sigma_r = 8;

    // Downsample grid
    Expr normalization = 1.0f / (cast<float>(sigma_s) * cast<float>(sigma_s));
    RDom rgrid(0, sigma_s, 0, sigma_s);

    Expr guide_pos = clamp(
        f_guide(x*sigma_s + rgrid.x, y*sigma_s + rgrid.y, n)*cast<float>(sigma_r),
        0, cast<float>(sigma_r));
    Expr lower_bin = cast<int>(floor(guide_pos));
    Expr upper_bin = cast<int>(ceil(guide_pos));
    Expr w = guide_pos - lower_bin;

    Func f_grid("f_grid");
    f_grid(x, y, z, ci, n) = 0.f;
    f_grid(x, y, lower_bin, ci, n) += 
        normalization*(1.0f-w)
      * f_input(x*sigma_s + rgrid.x, y*sigma_s + rgrid.y, ci, n);
    f_grid(x, y, upper_bin, ci, n) += 
        normalization*w
      * f_input(x*sigma_s + rgrid.x, y*sigma_s + rgrid.y, ci, n);

    // Perform 3D filtering in the grid
    Expr kw = filter.dim(0).extent();
    Expr kh = filter.dim(1).extent();
    Expr kd = filter.dim(2).extent();
    Expr in_chans = filter.dim(3).extent();
    RDom r(0, kw, 0, kh, 0, kd, 0, in_chans);
    Func f_conv("conv");
    f_conv(x, y, z, co, n)  = 0.f;
    f_conv(x, y, z, co, n) += 
      f_filter(r[0], r[1], r[2], r[3], co)
      * f_grid(x + r[0] - kw/2, 
               y + r[1] - kh/2,
               z + r[2] - kd/2,
               r[3], n);

    // Enclosing voxel
    Expr gx = (x+0.5f)/(1.0f*sigma_s);
    Expr gy = (y+0.5f)/(1.0f*sigma_s);
    Expr gz = clamp(f_guide(x, y, n)*cast<float>(sigma_r), 0.f, cast<float>(sigma_r));
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
    Func output("output");
    output(x, y, co, n) = 0.0f;
    output(x, y, co, n) +=
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
    func_map["output"] = output;

    return func_map;
}

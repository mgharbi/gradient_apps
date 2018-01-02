#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), c("c");

template <typename Input>
std::map<std::string, Func> bilateral_grid(
        const Input &input,
        const Input &filter_s,
        const Input &filter_r) {
    int sigma_s = 4;
    int sigma_r = 8;

    Func f_input("f_input");
    f_input(x, y, c) =
      Halide::BoundaryConditions::repeat_edge(input)(x, y, c);
    Func f_filter_s("f_filter_s");
    f_filter_s(x) = filter_s(x);
    Func f_filter_r("f_filter_r");
    f_filter_r(x) = filter_r(x);

    // Maybe also learn the luminance parameters?
    Func guide("guide");
    guide(x, y) = f_input(x, y, 0) * 0.299f +
                  f_input(x, y, 1) * 0.587f +
                  f_input(x, y, 2) * 0.114f;

    // Downsample grid
    RDom rgrid(0, sigma_s, 0, sigma_s);
    Expr guide_pos = clamp(guide(x * sigma_s + rgrid.x,
                                 y * sigma_s + rgrid.y) * cast<float>(sigma_r),
                           0.f,
                           cast<float>(sigma_r));
    Expr lower_bin = cast<int>(floor(guide_pos));
    Expr upper_bin = cast<int>(ceil(guide_pos));
    Expr w = guide_pos - lower_bin;

    Expr val = select(c < input.channels(),
                      f_input(cast<int>(x * sigma_s + rgrid.x),
                              cast<int>(y * sigma_s + rgrid.y),
                              c),
                      1.f) / cast<float>(sigma_s * sigma_s);
    Func f_grid("f_grid");
    f_grid(x, y, z, c) = 0.f;
    f_grid(x, y, lower_bin, c) += val * (1.f - w);
    f_grid(x, y, upper_bin, c) += val * w;

    // Perform 3D filtering in the grid
    RDom rr(filter_r);
    RDom rs(filter_s);
    Func blur_z("blur_z");
    blur_z(x, y, z, c) = 0.f;
    blur_z(x, y, z, c) += f_grid(x, y, z + rr.x - filter_r.width() / 2, c) *
                          abs(f_filter_r(rr.x));
    Func blur_y("blur_y");
    blur_y(x, y, z, c) = 0.f;
    blur_y(x, y, z, c) += blur_z(x, y + rs.x - filter_s.width() / 2, z, c) *
                          abs(f_filter_s(rs.x));
    Func blur_x("blur_x");
    blur_x(x, y, z, c) = 0.f;
    blur_x(x, y, z, c) += blur_y(x + rs.x - filter_s.width() / 2, y, z, c) *
                          abs(f_filter_s(rs.x));

    // Enclosing voxel
    Expr gx = x / float(sigma_s);
    Expr gy = y / float(sigma_s);
    Expr gz = clamp(guide(x, y) * cast<float>(sigma_r),
                    0.f,
                    cast<float>(sigma_r));
    Expr fx = cast<int>(floor(gx));
    Expr fy = cast<int>(floor(gy));
    Expr fz = cast<int>(floor(gz));
    Expr cx = fx + 1;
    Expr cy = fy + 1;
    Expr cz = cast<int>(ceil(gz));
    Expr wx = gx - fx;
    Expr wy = gy - fy;
    Expr wz = gz - fz;

    // trilerp
    Func unnormalized_output("unnormalized_output");
    unnormalized_output(x, y, c) =
         blur_x(fx, fy, fz, c)*(1.f - wx)*(1.f - wy)*(1.f - wz)
       + blur_x(fx, fy, cz, c)*(1.f - wx)*(1.f - wy)*(      wz)
       + blur_x(fx, cy, fz, c)*(1.f - wx)*(      wy)*(1.f - wz)
       + blur_x(fx, cy, cz, c)*(1.f - wx)*(      wy)*(      wz)
       + blur_x(cx, fy, fz, c)*(      wx)*(1.f - wy)*(1.f - wz)
       + blur_x(cx, fy, cz, c)*(      wx)*(1.f - wy)*(      wz)
       + blur_x(cx, cy, fz, c)*(      wx)*(      wy)*(1.f - wz)
       + blur_x(cx, cy, cz, c)*(      wx)*(      wy)*(      wz);
    Func output("output");
    output(x, y, c) = unnormalized_output(x, y, c) /
                      (unnormalized_output(x, y, input.channels()) + 1e-4f);

    std::map<std::string, Func> func_map;
    func_map["f_input"]  = f_input;
    func_map["f_filter_s"] = f_filter_s;
    func_map["f_filter_r"] = f_filter_r;
    func_map["output"] = output;
    return func_map;
}


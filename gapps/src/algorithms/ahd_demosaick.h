#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), c("c");

template <typename Input>
std::map<std::string, Func> ahd_demosaick(
        const Input &mosaick) {
    // Based on AmaZE: https://github.com/LibRaw/LibRaw-demosaic-pack-GPL3/blob/master/amaze_demosaic_RT.cc
    Func f_mosaick("f_mosaick");
    f_mosaick(x, y) = Halide::BoundaryConditions::constant_exterior(
        mosaick, 0.0f)(x, y);

    float eps = 1e-5;
    float arthresh = 0.75; // adaptive ratio threshold

    // Gradients
    Func grad_h("grad_h");
    Func grad_v("grad_v");
    Func grad_p("grad_p");
    Func grad_m("grad_m");
    grad_h(x, y) = abs(f_mosaick(x+1, y) - f_mosaick(x-1, y));
    grad_v(x, y) = abs(f_mosaick(x, y+1) - f_mosaick(x, y-1));
    grad_p(x, y) = abs(f_mosaick(x+1, y-1) - f_mosaick(x-1, y+1));
    grad_m(x, y) = abs(f_mosaick(x+1, y+1) - f_mosaick(x-1, y-1));

    // directional weights for interpolation
    Func dir_w("dir_w");
    dir_w(x, y, c) = 0.0f;
    dir_w(x, y, 0) = eps + grad_v(x, y-1) + grad_v(x, y) + grad_v(x, y+1);
    dir_w(x, y, 1) = eps + grad_h(x-1, y) + grad_h(x, y) + grad_h(x+1, y);
    // Dgrbpsq
    // Dgrbmsq
    
    // Color ratios (up, down, left, right)
    Func cr_u("cr_u");
    Func cr_d("cr_d");
    Func cr_l("cr_l");
    Func cr_r("cr_r");
    cr_u(x, y) = f_mosaick(x, y-1)*(dir_w(x, y-2, 0) + dir_w(x, y, 0)) / 
      (dir_w(x, y-2, 0)*(eps + f_mosaick(x, y)) + dir_w(x,y, 0)*(eps + f_mosaick(x, y-2)));
    cr_d(x, y) = f_mosaick(x, y+1)*(dir_w(x, y+2, 0) + dir_w(x, y, 0)) / 
      (dir_w(x, y+2, 0)*(eps + f_mosaick(x, y)) + dir_w(x,y, 0)*(eps + f_mosaick(x, y+2)));
    cr_l(x, y) = f_mosaick(x-1, y)*(dir_w(x-2, y, 1) + dir_w(x, y, 1)) / 
      (dir_w(x-2, y, 1)*(eps + f_mosaick(x, y)) + dir_w(x, y, 1)*(eps + f_mosaick(x-2, y)));
    cr_r(x, y) = f_mosaick(x+1, y)*(dir_w(x+2, y, 1) + dir_w(x, y, 1)) / 
      (dir_w(x+2, y, 1)*(eps + f_mosaick(x, y)) + dir_w(x, y, 1)*(eps + f_mosaick(x+2, y)));
    
    // Interpolate G using Hamilton-Adams []
    Func g_ha_u("g_ha_u");
    Func g_ha_d("g_ha_d");
    Func g_ha_l("g_ha_l");
    Func g_ha_r("g_ha_r");
    g_ha_u(x, y) = f_mosaick(x, y-1) + 0.5f*(f_mosaick(x, y) - f_mosaick(x, y-2));
    g_ha_d(x, y) = f_mosaick(x, y+1) + 0.5f*(f_mosaick(x, y) - f_mosaick(x, y+2));
    g_ha_l(x, y) = f_mosaick(x-1, y) + 0.5f*(f_mosaick(x, y) - f_mosaick(x-2, y));
    g_ha_r(x, y) = f_mosaick(x+1, y) + 0.5f*(f_mosaick(x, y) - f_mosaick(x+2, y));
    // TODO: clip
    
    // Interpolate G using color ratios
    Func g_ar_u("g_ar_u");
    Func g_ar_d("g_ar_d");
    Func g_ar_l("g_ar_l");
    Func g_ar_r("g_ar_r");
    g_ar_u(x, y) = select(1.0f - cr_u(x, y) < arthresh, f_mosaick(x, y)*cr_u(x, y), g_ha_u(x,y));
    g_ar_d(x, y) = select(1.0f - cr_d(x, y) < arthresh, f_mosaick(x, y)*cr_d(x, y), g_ha_d(x,y));
    g_ar_l(x, y) = select(1.0f - cr_l(x, y) < arthresh, f_mosaick(x, y)*cr_l(x, y), g_ha_l(x,y));
    g_ar_r(x, y) = select(1.0f - cr_r(x, y) < arthresh, f_mosaick(x, y)*cr_r(x, y), g_ha_r(x,y));
    
    Expr v_w = dir_w(x, y-1, 0) / (dir_w(x, y-1, 0) + dir_w(x, y+1, 0));
    Expr h_w = dir_w(x-1, y, 1) / (dir_w(x-1, y, 1) + dir_w(x+1, y, 1));

    // Interpolate G
    Func g_int_ar_v("g_int_ar_v");
    Func g_int_ar_h("g_int_ar_h");
    Func g_int_ha_v("g_int_ha_v");
    Func g_int_ha_h("g_int_ha_h");
    g_int_ar_v(x, y) = g_ar_d(x, y)*v_w + (1.0f - v_w)*g_ar_u(x, y);
    g_int_ar_h(x, y) = g_ar_r(x, y)*h_w + (1.0f - h_w)*g_ar_l(x, y);
    g_int_ha_v(x, y) = g_ha_d(x, y)*v_w + (1.0f - v_w)*g_ha_u(x, y);
    g_int_ha_h(x, y) = g_ha_r(x, y)*h_w + (1.0f - h_w)*g_ha_l(x, y);
    
    Func sign("sign");
    sign(x, y) = select( 
        (y % 2 == 0 && x % 2 == 0) || (y % 2 == 1 && x % 2 == 1), -1.0f, // green sites
        1.0f); // R, B sites

    // Interpolate color differences
    Func cd_v("cd_v");
    Func cd_h("cd_h");
    Func cd_v_alt("cd_v_alt");
    Func cd_h_alt("cd_h_alt");
    cd_v(x, y) = sign(x, y)*(g_int_ar_v(x, y) - f_mosaick(x, y));
    cd_h(x, y) = sign(x, y)*(g_int_ar_h(x, y) - f_mosaick(x, y));
    cd_v_alt(x, y) = sign(x, y)*(g_int_ha_v(x, y) - f_mosaick(x, y));
    cd_h_alt(x, y) = sign(x, y)*(g_int_ha_h(x, y) - f_mosaick(x, y));
    
    // TODO if above clipping point, rever to HA l.511

    // Differences of interpolations
    Func diff_g_int_v("diff_g_int_v");
    Func diff_g_int_h("diff_g_int_h");
    diff_g_int_v(x, y) = min(pow(g_ha_u(x, y) - g_ha_d(x, y), 2), pow(g_ar_u(x, y) - g_ar_d(x, y), 2));
    diff_g_int_h(x, y) = min(pow(g_ha_l(x, y) - g_ha_r(x, y), 2), pow(g_ar_l(x, y) - g_ar_r(x, y), 2));

    // Variance of the color interp
    Func cd_h_var("cd_h_var");
    Func cd_v_var("cd_v_var");
    Func cd_h_alt_var("cd_h_alt_var");
    Func cd_v_alt_var("cd_v_alt_var");
    cd_h_var(x, y) = 3.0f*(pow(cd_h(x-2, y), 2) + pow(cd_h(x, y), 2) + pow(cd_h(x+2, y), 2)) -
      pow(cd_h(x-2, y) + cd_h(x, y) + cd_h(x+2, y), 2);
    cd_h_alt_var(x, y) = 3.0f*(pow(cd_h_alt(x-2, y), 2) + pow(cd_h_alt(x, y), 2) + pow(cd_h_alt(x+2, y), 2)) -
      pow(cd_h_alt(x-2, y) + cd_h_alt(x, y) + cd_h_alt(x+2, y), 2);
    cd_v_var(x, y) = 3.0f*(pow(cd_v(x, y-2), 2) + pow(cd_v(x, y), 2) + pow(cd_v(x, y+2), 2)) -
      pow(cd_v(x, y-2) + cd_v(x, y) + cd_v(x, y+2), 2);
    cd_v_alt_var(x, y) = 3.0f*(pow(cd_v_alt(x, y-2), 2) + pow(cd_v_alt(x, y), 2) + pow(cd_v_alt(x, y+2), 2)) -
      pow(cd_v_alt(x, y-2) + cd_v_alt(x, y) + cd_v_alt(x, y+2), 2);

    // Pick the interpolation with smallest variance
    Func cd_merge_v("cd_merge_v");
    Func cd_merge_h("cd_merge_h");
    cd_merge_v(x, y) = select(cd_v_alt_var < cd_v_var, cd_v_alt(x, y), cd_v(x, y));
    cd_merge_h(x, y) = select(cd_h_alt_var < cd_h_var, cd_h_alt(x, y), cd_h(x, y));

    Func f_output("f_output");
    f_output(x, y, c) = g_int_ar_h(x,y);
    // f_output(x, y, c) = select(
    //     c == 0, 0.0f,
    //     c == 1, 0.0f,
    //     0.0f);

    std::map<std::string, Func> func_map;
    func_map["mosaick"]  = f_mosaick;
    func_map["output"]  = f_output;
    return func_map;
}

// template <typename Input>
// std::map<std::string, Func> ahd_demosaick(
//         const Input &mosaick) {
//     Func f_mosaick("f_mosaick");
//     f_mosaick(x, y) = Halide::BoundaryConditions::constant_exterior(
//         mosaick, 0.0f)(x, y);
//
//     Func f_g0("f_g0");
//     Func f_r1("f_r1");
//     Func f_b2("f_b2");
//     Func f_g3("f_g3");
//     // f_g0(x, y) = f_mosaick(2*x, 2*y);
//     f_r1(x, y)  = f_mosaick(2*x+1, 2*y);
//     // f_b2(x, y)  = f_mosaick(2*x, 2*y+1);
//     // f_g3(x, y) = f_mosaick(2*x+1, 2*y+1);
//
//     Halide::Buffer<float> g1_filter(5);
//     g1_filter(0) = -0.2569;
//     g1_filter(1) = 0.4339;
//     g1_filter(2) = 0.5138;
//     g1_filter(3) = 0.4339;
//     g1_filter(4) = -0.2569;
//
//     Func f_g1_h("f_g1_h"); // at red locations
//     RDom r5(-2, 5);
//
//     // Combine to form green channel, interpolated horizontally
//     Func f_g_h("f_g_h");
//     f_g_h(x, y) = select(
//         y % 2 == 0 && x == 1, sum(f_mosaick(x + r5, y)*g1_filter(r5+2)),
//         x % 2 == 0 && y % 2 == 1, 0, sum(f_mosaick(x + r5, y)*g1_filter(r5+2)),
//         f_mosaick(x, y));
//     
//     // Combine to form green channel, interpolated vertically
//     Func f_g_v("f_g");
//     f_g_v(x, y) = select(
//         y % 2 == 0 && x == 1, sum(f_mosaick(x + r5, y)*g1_filter(r5+2)),
//         x % 2 == 0 && y % 2 == 1, 0, sum(f_mosaick(x + r5, y)*g1_filter(r5+2)),
//         f_mosaick(x, y));
//
//     // Difference with reconstructed G
//     Func f_dg_h("f_dg_h");
//     f_dg_h(x, y) = f_mosaick(x, y) - f_g_h(x, y);
//     Func f_dg_v("f_dg_v");
//     f_dg_v(x, y) = f_mosaick(x, y) - f_g_v(x, y);
//
//     // Interpolate R-G and construct R
//     Func f_r_h("f_r_h");
//     f_r_h(x, y) = select(
//         x % 2 == 0 && y % 2 == 0, 
//           (f_dg_h(x-1, y) + f_dg_h(x+1, y))*0.5f + f_g_h(x, y),
//         x % 2 == 0 && y % 2 == 1, 
//           (f_dg_h(x - 1, y - 1) + f_dg_h(x - 1, y + 1) + 
//           f_dg_h(x + 1, y - 1) + f_dg_h(x + 1, y + 1))*0.25f + f_g_h(x, y),
//         x % 2 == 1 && y % 2 == 1, (
//           f_dg_h(x, y - 1) + f_dg_h(x, y + 1))*0.5f + f_g_h(x, y),
//         f_mosaick(x, y)); // known
//     
//     // Interpolate B-G and construct B
//     Func f_b_h("f_b_h");
//     f_b_h(x, y) = select(
//         x % 2 == 0 && y % 2 == 0, 
//           (f_dg_h(x, y-1) + f_dg_h(x, y+1))*0.5f + f_g_h(x, y),
//         x % 2 == 1 && y % 2 == 0, 
//           (f_dg_h(x - 1, y - 1) + f_dg_h(x - 1, y + 1) +
//            f_dg_h(x + 1, y - 1) + f_dg_h(x + 1, y + 1))*0.25f + f_g_h(x, y),
//         x % 2 == 1 && y % 2 == 1, 
//           (f_dg_h(x - 1, y) + f_dg_h(x + 1, y))*0.5f + f_g_h(x, y),
//         f_mosaick(x, y)); // known
//     // -----------------------------------------------------------------------
//
//     Func f_output("f_output");
//     f_output(x, y, c) = select(
//         c == 0, 0.0f, // f_r_h(x, y),
//         c == 1, f_g_h(x, y),
//         0.0f);//f_b_h(x, y));
//
//     std::map<std::string, Func> func_map;
//     func_map["mosaick"]  = f_mosaick;
//     func_map["output"]  = f_output;
//     return func_map;
// }

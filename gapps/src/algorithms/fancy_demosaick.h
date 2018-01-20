#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

#define n_w 6
#define n_w2 4

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n"), k("k");

template <typename Input, typename InputArray, typename InputArray2>
std::map<std::string, Func> fancy_demosaick(
        const Input &cfa,
        const InputArray &weights,
        const InputArray2 &weights2d
        ) {

    Func cfa_("cfa_");
    cfa_(x, y, n) = Halide::BoundaryConditions::constant_exterior(
        cfa, 0.0f)(x, y, n);

    Expr is_green = (x % 2 == y % 2);
    Expr is_g0 = (x % 2 == 0) && (y % 2 == 0);
    Expr is_g3 = (x % 2 == 1) && (y % 2 == 1);
    Expr is_red = (x % 2 == 1) && (y % 2 == 0);
    Expr is_blue = (x % 2 == 0) && (y % 2 == 1);

    float ar_thresh = 0.75;
    float eps = 1e-5;

    // Compute image gradients
    Func grads("grads");
    grads(x, y, k, n) = select(
        k == 0, abs(cfa_(x+1, y, n) - cfa_(x-1, y, n)),  // dx
        k == 1, abs(cfa_(x, y+1, n) - cfa_(x, y-1, n)),  // dy
        k == 2, abs(cfa_(x+1, y+1, n) - cfa_(x-1, y-1, n)),  // south east
        k == 3, abs(cfa_(x+1, y-1, n) - cfa_(x-1, y+1, n)),  // north east
        0.0f);

    // Directional weights
    Func dir_wts_x("dir_wts_x");
    Func dir_wts_y("dir_wts_y");
    RDom r_dir_wts(-2, 5);
    dir_wts_x(x, y, n) = eps;
    dir_wts_x(x, y, n) += grads(x + r_dir_wts, y, 0, n)*weights[0](2+r_dir_wts);
    dir_wts_y(x, y, n) = eps;
    dir_wts_y(x, y, n) += grads(x, y + r_dir_wts, 1, n)*weights[1](2+r_dir_wts);

    // . G .    . 3 .
    // G . G    2 . 0
    // . G .    . 1 .
    RDom rN(0, 4, "rN");
    Expr dx = select(rN == 0, 1, rN == 2, -1, 0);
    Expr dy = select(rN == 1, 1, rN == 3, -1, 0);
    Expr k_dx = select(k == 0, 1, k == 2, -1, 0);
    Expr k_dy = select(k == 1, 1, k == 3, -1, 0);

    // ---- H/V interpolation of green at R/B ---------------------------------
    //
    // Weights on the four green neighbors, based on gradients terms
    Func green_weights("green_weights");
    green_weights(x, y, k, n) = 0.0f;
    green_weights(x, y, k, n) += 
        dir_wts_x(x+dx, y+dy, n)*weights2d[0](rN, k)
      + dir_wts_y(x+dx, y+dy, n)*weights2d[1](rN, k);

    // Cardinal interpolations
    Func c_g("c_g");
    RDom ri(-1, 4, "ri");
    c_g(x, y, k, n) = 0.0f;
    c_g(x, y, k, n) += cfa_(x + ri*k_dx, y + ri*k_dy, n)*weights[2](1+ri);

    // Interpolate green
    Func i_g("i_g");
    i_g(x, y, n) = 0.0f;
    i_g(x, y, n) += c_g(x, y, rN, n)*green_weights(x, y, rN, n);

    // Normalize interpolation
    Func gw_sum("gw_sum");
    gw_sum(x, y, n) = 0.0f;
    gw_sum(x, y, n) += green_weights(x, y, rN, n);
    Func n_i_g("n_i_g");
    n_i_g(x, y, n) = i_g(x, y, n) / gw_sum(x, y, n);

    // Interpolated green
    Func g("g");
    g(x, y, n) = select(
        is_green, cfa_(x, y, n),
        n_i_g(x, y, n));

    // ------------------------------------------------------------------------

    // Color difference, on separate channels
    Func cd("cd"); 
    cd(x, y, n) = cfa(x, y, n) - g(x, y, n);

    // ---- Diagonal interpolation of R at B (and vice versa) -----------------
    
    Func cd_grads("cd_grads");
    cd_grads(x, y, k, n) = select(
        k == 0, abs(cd(x+1, y+1, n) - cd(x-1, y-1, n)),  // south east
        k == 1, abs(cd(x+1, y-1, n) - cd(x-1, y+1, n)),  // north east
        0.0f);

    // Diagonal gradient weights
    r_dir_wts = RDom(-2, 5);
    Func dir_wts_p("dir_wts_p");
    Func dir_wts_n("dir_wts_n");
    dir_wts_p(x, y, n) = eps;
    dir_wts_p(x, y, n) += cd_grads(x + r_dir_wts, y + r_dir_wts, 0, n)*weights[3](2+r_dir_wts);
    dir_wts_n(x, y, n) = eps;
    dir_wts_n(x, y, n) += cd_grads(x + r_dir_wts, y + r_dir_wts, 1, n)*weights[4](2+r_dir_wts);

    // B . B    3 . 0
    // . R .    . . .
    // B . B    2 . 1
    RDom rD(0, 4, "rD");
    Expr dx_diag = select(rD == 0 || rD == 1, 1, -1);
    Expr dy_diag = select(rD == 0 || rD == 3, -1, 1);
    Expr k_dx_diag = select(k == 0 || k == 1, 1, 1);
    Expr k_dy_diag = select(k == 0 || k == 3, -1, 1);

    Func diag_weights("diag_weights");
    diag_weights(x, y, k, n) = 0.0f;
    diag_weights(x, y, k, n) += 
        dir_wts_p(x+dx_diag, y+dy_diag, n)*weights2d[2](rD, k)
      + dir_wts_n(x+dx_diag, y+dy_diag, n)*weights2d[3](rD, k);
    
    // Diagonal interpolations, for northeast dir, weight the four capital B
    //     B   B
    //     .
    // b . B . B 
    // . R . 
    // b . b 
    Func d_rb("d_rb");
    ri = RDom(0, 4, "ri");
    d_rb(x, y, k, n) = 0.0f;
    d_rb(x, y, k, n) += cd(x + ri*k_dx_diag, y + ri*k_dy_diag, n)*weights[5](ri-1);

    // Interpolate color difference
    Func i_cd("i_cd");
    i_cd(x, y, n) = 0.0f;
    i_cd(x, y, n) += d_rb(x, y, rD, n)*diag_weights(x, y, rD, n);

    // Normalize interpolation
    Func rbw_sum("rbw_sum");
    rbw_sum(x, y, n) = 0.0f;
    rbw_sum(x, y, n) += diag_weights(x, y, rD, n);
    Func n_i_cd("n_i_cd");
    n_i_cd(x, y, n) = i_cd(x, y, n) / rbw_sum(x, y, n);

    // Split color difference over two channels
    Func split_cd("split_cd");
    split_cd(x, y, k, n) = select(
        k == 0 && is_blue, n_i_cd(x, y, n), // interpolated red
        k == 1 && is_red,  n_i_cd(x, y, n), // interpolated blue
        cd(x, y, n)); // known red, known blue or green with no data
    // ------------------------------------------------------------------------

    // ---- H/V interpolation of color difference at green --------------------
    
    Func final_cd("final_cd");
    final_cd(x, y, k, n) = select(
        is_green, 0.0f, split_cd(x, y, k, n));

    // ------------------------------------------------------------------------
    
    Func output("output");
    output(x, y, c, n) = g(x, y, n) + select(
        c == 0, final_cd(x, y, 0, n),
        c == 2, final_cd(x, y, 1, n),
        0.0f);

    std::map<std::string, Func> func_map;
    func_map["output"] = output;

    return func_map;
}

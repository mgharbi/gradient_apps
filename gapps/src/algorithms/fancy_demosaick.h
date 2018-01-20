#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

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
    RDom r_dir_wts(-1, 3);
    dir_wts_x(x, y, n) = eps;
    dir_wts_x(x, y, n) += grads(x + r_dir_wts, y, 0, n)*weights[0](1+r_dir_wts);
    dir_wts_y(x, y, n) = eps;
    dir_wts_y(x, y, n) += grads(x, y + r_dir_wts, 1, n)*weights[1](1+r_dir_wts);

    // // . G .
    // // G . G
    // // . G .
    RDom rN(0, 4, "rN");
    Expr dx = select(rN == 0, 1, rN == 2, -1, 0);
    Expr dy = select(rN == 1, 1, rN == 3, -1, 0);


    // Weights on the four green neighbors, based on gradients
    Func green_weights("green_weights");
    green_weights(x, y, k, n) = 0.0f;
    green_weights(x, y, k, n) += 
        dir_wts_x(x+dx, y+dy, n)*weights2d[0](rN, k)
      + dir_wts_y(x+dx, y+dy, n)*weights2d[1](rN, k);

    Func gw_sum("gw_sum");
    gw_sum(x, y, n) = 0.0f;
    gw_sum(x, y, n) += green_weights(x, y, rN, n);

    // Interpolate green
    Func i_g("i_g");
    // i_g(x, y, n) = gw_sum(x, y, n);
    i_g(x, y, n) = 0.0f;
    i_g(x, y, n) += cfa_(x+dx, y+dy, n)*green_weights(x, y, rN, n);
    i_g(x, y, n) /= gw_sum(x, y, n);

    Func g("g");
    g(x, y, n) = select(
        is_green, cfa_(x, y, n),
        i_g(x, y, n));

    Func output("output");
    output(x, y, c, n) = select(
        c == 1, g(x, y, n),
        0.0f);

    std::map<std::string, Func> func_map;
    func_map["output"] = output;

    return func_map;
}

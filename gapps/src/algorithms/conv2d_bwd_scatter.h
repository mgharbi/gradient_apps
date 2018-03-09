#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), n("n"), ci("ci"), co("co");

template <typename Input>
std::map<std::string, Func> conv2d_bwd_scatter(
        const Input &d_output,
        const Input &filter) {

    // Wrap inputs in Funcs
    Func f_d_output("f_d_output");
    f_d_output(x, y, co, n) = 
      Halide::BoundaryConditions::repeat_edge(d_output)(x, y, co, n);
    Func f_filter("f_filter");
    f_filter(x, y, ci, co) = filter(x, y, ci, co);

    Expr width = d_output.dim(0).extent();
    Expr height = d_output.dim(1).extent();

    // Perform 2d filtering in the grid
    Expr kw = filter.dim(0).extent();
    Expr kh = filter.dim(1).extent();
    Expr in_chans = filter.dim(2).extent();
    Expr start_w = -cast<int>(kw/2);
    Expr start_h = -cast<int>(kh/2);
    RDom r(start_w, kw, start_h, kh, 0, in_chans, 0, width, 0, height);
    Func f_d_input("f_d_input");
    f_d_input(x, y, ci, n)  = 0.f;
    f_d_input(r[3] + r[0], r[4] + r[1], ci, n)  += 
        f_filter(r[0] - start_w, r[1] - start_h, ci, r[2])
      * f_d_output(r[3], r[4], r[2], n);

    std::map<std::string, Func> func_map;
    func_map["d_output"]  = f_d_output;
    func_map["filter"] = f_filter;
    func_map["d_input"] = f_d_input;

    return func_map;
}

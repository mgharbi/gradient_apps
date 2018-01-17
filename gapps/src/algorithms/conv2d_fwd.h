#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), n("n"), ci("ci"), co("co");

template <typename Input>
std::map<std::string, Func> conv2d_fwd(
        const Input &input,
        const Input &filter) {

    // Wrap inputs in Funcs
    Func f_input("f_input");
    f_input(x, y, ci, n) = 
      Halide::BoundaryConditions::repeat_edge(input)(x, y, ci, n);
    Func f_filter("f_filter");
    f_filter(x, y, ci, co) = filter(x, y, ci, co);

    // Perform 2d filtering in the grid
    Expr kw = filter.dim(0).extent();
    Expr kh = filter.dim(1).extent();
    Expr in_chans = filter.dim(1).extent();
    RDom r(0, kw, 0, kh, 0, in_chans);
    Func f_output("f_output");
    f_output(x, y, co, n)  = 0.f;
    f_output(x, y, co, n) += 
      f_filter(r[0], r[1], r[2], co)
      * f_input(x + r[0] - kw/2, y + r[1] - kh/2, r[2], n);

    std::map<std::string, Func> func_map;
    func_map["input"]  = f_input;
    func_map["filter"] = f_filter;
    func_map["output"] = f_output;

    return func_map;
}

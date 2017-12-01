#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), n("n"), ci("ci"), co("co");

template <typename Input>
std::map<std::string, Func> conv1d(
        const Input &input,
        const Input &filter) {

    // Wrap inputs in Funcs
    Func f_input("f_input");
    f_input(x, ci, n) = 
      Halide::BoundaryConditions::repeat_edge(input)(x, ci, n);
    Func f_filter("f_filter");
    f_filter(x, ci, co) = filter(x, ci, co);

    // Perform 1d filtering in the grid
    Expr kw = filter.dim(0).extent();
    Expr in_chans = filter.dim(1).extent();
    RDom r(0, kw, 0, in_chans);
    Func f_output("f_output");
    f_output(x, co, n)  = 0.f;
    f_output(x, co, n) += f_filter(r[0], r[1], co) *
                                f_input(x + r[0] - kw/2, r[1], n);

    std::map<std::string, Func> func_map;
    func_map["input"]  = f_input;
    func_map["filter"] = f_filter;
    func_map["output"] = f_output;

    return func_map;
}

#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), n("n"), ci("ci"), co("co");

template <typename Input>
std::map<std::string, Func> conv3d(
        const Input &input,
        const Input &filter) {

    // Wrap inputs in Funcs
    Func f_input("f_input");
    f_input(x, y, z, ci, n) = 
      Halide::BoundaryConditions::repeat_edge(input)(x, y, z, ci, n);
    Func f_filter("f_filter");
    f_filter(x, y, z, ci, co) = filter(x, y, z, ci, co);

    // Perform 3D filtering in the grid
    Expr kw = filter.dim(0).extent();
    Expr kh = filter.dim(1).extent();
    Expr kd = filter.dim(2).extent();
    Expr in_chans = filter.dim(3).extent();
    RDom r(0, kw, 0, kh, 0, kd, 0, in_chans);
    Func f_output("f_output");
    f_output(x, y, z, co, n)  = 0.f;
    f_output(x, y, z, co, n) += f_filter(r[0], r[1], r[2], r[3], co) *
                                f_input(x + r[0] - kw/2, 
                                       y + r[1] - kh/2,
                                       z + r[2] - kd/2,
                                       r[3], n);

    std::map<std::string, Func> func_map;
    func_map["input"]  = f_input;
    func_map["filter"] = f_filter;
    func_map["output"] = f_output;

    return func_map;
}

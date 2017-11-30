#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y");

template <typename Input>
std::map<std::string, Func> soft_histogram(
        const Input &input, const Expr nbins) {
    Func f_input("f_input");
    f_input(x, y) = Halide::BoundaryConditions::constant_exterior(
        input, 0.0f)(x, y);

    RDom r(0, input.dim(0).extent(), 0, input.dim(1).extent());

    Expr bin_pos = clamp(f_input(r.x, r.y), 0.0f, 1.0f)*(nbins-1);
    Expr lower_bin = max(cast<int>(floor(bin_pos)), 0);
    Expr upper_bin = min(cast<int>(ceil(bin_pos)), nbins-1);

    Expr w = bin_pos  - lower_bin;
    Func f_output("f_output");
    f_output(x) = 0.0f;
    f_output(lower_bin) += 1.0f - w;
    f_output(upper_bin) += w;

    std::map<std::string, Func> func_map;
    func_map["input"]  = f_input;
    func_map["output"]  = f_output;
    return func_map;
}

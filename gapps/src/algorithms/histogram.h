#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y");

template <typename Input>
std::map<std::string, Func> histogram(
        const Input &input, const Expr nbins) {
    Func f_input("f_input");
    f_input(x, y) = Halide::BoundaryConditions::constant_exterior(
        input, 0.0f)(x, y);

    RDom r(0, input.dim(0).extent(), 0, input.dim(1).extent());

    Func f_output("f_output");
    f_output(x, y) = 0.0f;
    f_output(r.x, r.y) = f_input(r.x, r.y);
    // Expr bin = cast<int>(clamp(f_input(r.x, r.y), 0.0f, 1.0f)*(nbins-1));
    // f_output(bin) += 1.0f;

    std::map<std::string, Func> func_map;
    func_map["input"]  = f_input;
    func_map["output"]  = f_output;
    return func_map;
}

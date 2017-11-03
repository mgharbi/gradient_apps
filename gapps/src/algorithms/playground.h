#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), n("n"), ci("ci"), co("co");

template <typename Input>
std::map<std::string, Func> playground(
        const Input &input1,
        const Input &input2) {
    // Func f_input1("f_input1");
    // f_input1(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
    //     input1, 0.0f)(x, y, ci, n);
    // Func f_input2("f_input2");
    // f_input2(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
    //   input2, 0.0f)(x, y, ci, n);

    Func f_input1("f_input1");
    f_input1(x, y, ci, n) = (input1)(x, y, ci, n);
    Func f_input2("f_input2");
    f_input2(x, y, ci, n) = (input2)(x, y, ci, n);

    Func f_output("f_output");
    Expr sigma = input2.dim(0).extent()-1;
    Expr guide_pos = clamp(f_input1(x, y, ci, n)*sigma, 0, cast<float>(sigma));
    Expr lower_bin = cast<int>(floor(guide_pos));
    f_output(x, y, ci, n) = f_input2(lower_bin, y, ci, n);

    // Func f_output("f_output");
    // RDom r(0, 2);
    // f_output(x, y, ci, n) = f_input2(x, y, ci, n);
    // f_output(x, y, ci, n) += f_input1(x + r.x, y, ci, n);

    std::map<std::string, Func> func_map;
    func_map["output"]  = f_output;
    func_map["input1"]  = f_input1;
    func_map["input2"]  = f_input2;
    return func_map;
}

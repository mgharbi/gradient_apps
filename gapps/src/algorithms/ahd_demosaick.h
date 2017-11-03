#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), c("c");

template <typename Input>
std::map<std::string, Func> ahd_demosaick(
        const Input &mosaick) {
    Func f_mosaick("f_mosaick");
    f_mosaick(x, y) = Halide::BoundaryConditions::constant_exterior(
        mosaick, 0.0f)(x, y);

    Func f_g0("f_g0");
    Func f_r("f_r");
    Func f_b("f_b");
    Func f_g3("f_g3");
    f_g0(x, y) = f_mosaick(2*x, 2*y);
    f_r(x, y)  = f_mosaick(2*x+1, 2*y);
    f_b(x, y)  = f_mosaick(2*x, 2*y+1);
    f_g3(x, y) = f_mosaick(2*x+1, 2*y+1);

    Func f_g1("f_g1");
    RDom gr_green_h(0, 5);
    // f_g1(x, y) = f_mosaick(2*x, 2*y);

    Func f_output("f_output");
    f_output(x, y, c) = select(
        c == 0, f_r(x/2, y/2),
        c==1, f_g0(x/2, y/2),
        f_b(x/2, y/2));

    std::map<std::string, Func> func_map;
    func_map["mosaick"]  = f_mosaick;
    func_map["output"]  = f_output;
    return func_map;
}

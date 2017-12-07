#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), c("c");

template <typename Input>
std::map<std::string, Func> naive_demosaick(
        const Input &mosaick) {
    Func f_mosaick("f_mosaick");
    f_mosaick(x, y) = Halide::BoundaryConditions::repeat_edge(
        mosaick)(x, y);

    Expr is_green = (x % 2 == y % 2);
    Expr is_g0 = (x % 2 == 0) && (y % 2 == 0);
    Expr is_g3 = (x % 2 == 1) && (y % 2 == 1);
    Expr is_red = (x % 2 == 1) && (y % 2 == 0);
    Expr is_blue = (x % 2 == 0) && (y % 2 == 1);

    Expr up = f_mosaick(x, y-1);
    Expr down = f_mosaick(x, y+1);
    Expr left = f_mosaick(x-1, y);
    Expr right = f_mosaick(x+1, y);

    Expr diffv = abs(up-down);
    Expr diffh = abs(right-left);

    Func interpolated_green("interpolated_green");
    interpolated_green(x, y) = select(diffv < diffh, 0.5f*(up+down),
                                      0.5f*(left+right));

    Func green("green");
    green(x, y) = select(is_green, f_mosaick(x, y), interpolated_green(x, y));

    // -- Chroma interpolation ------------------------------------------------
    Func chroma("chroma");
    chroma(x, y) = f_mosaick(x, y) - green(x, y);

    Func h_interp("h_interp");
    h_interp(x, y) = 0.5f*(chroma(x-1, y) + chroma(x+1, y)) + green(x, y);

    Func v_interp("v_interp");
    v_interp(x, y) = 0.5f*(chroma(x, y-1) + chroma(x, y+1)) + green(x, y);

    Func q_interp("q_interp");  // quincux interpolation, when we have 4 neighbors
    q_interp(x, y) = 0.25f*(
        chroma(x-1, y-1) + 
        chroma(x-1, y+1) +
        chroma(x+1, y-1) +
        chroma(x+1, y+1)) + green(x, y);

    Func red("red");
    red(x, y) = select(is_g0, h_interp(x, y),
                       is_g3, v_interp(x, y),
                       is_blue, q_interp(x, y),
                       f_mosaick(x, y));

    Func blue("blue");
    blue(x, y) = select(is_g0, v_interp(x, y),
                        is_g3, h_interp(x, y),
                        is_red, q_interp(x, y),
                        f_mosaick(x, y));

    Func f_output("f_output");
    // f_output(x, y, c) = select(c == 0, red(x, y),
    //                            c == 1, green(x, y),
    //                            blue(x, y));
    f_output(x, y, c) = green(x, y);

    std::map<std::string, Func> func_map;
    func_map["mosaick"]  = f_mosaick;
    func_map["output"]  = f_output;

    return func_map;
}

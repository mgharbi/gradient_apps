#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> naive_demosaick(
        const Input &mosaick) {
    Func f_mosaick("f_mosaick");
    f_mosaick(x, y, n) = Halide::BoundaryConditions::repeat_edge(
        mosaick)(x, y, n);

    Expr is_green = (x % 2 == y % 2);
    Expr is_g0 = (x % 2 == 0) && (y % 2 == 0);
    Expr is_g3 = (x % 2 == 1) && (y % 2 == 1);
    Expr is_red = (x % 2 == 1) && (y % 2 == 0);
    Expr is_blue = (x % 2 == 0) && (y % 2 == 1);

    Expr up = f_mosaick(x, y-1, n);
    Expr down = f_mosaick(x, y+1, n);
    Expr left = f_mosaick(x-1, y, n);
    Expr right = f_mosaick(x+1, y, n);

    Expr diffv = abs(up-down);
    Expr diffh = abs(right-left);

    Func interpolated_green("interpolated_green");
    interpolated_green(x, y, n) = select(diffv < diffh, 0.5f*(up+down),
                                      0.5f*(left+right));

    Func green("green");
    green(x, y, n) = select(is_green, f_mosaick(x, y, n), interpolated_green(x, y, n));

    // -- Chroma interpolation ------------------------------------------------
    Func chroma("chroma");
    chroma(x, y, n) = f_mosaick(x, y, n) - green(x, y, n);

    Func h_interp("h_interp");
    h_interp(x, y, n) = 0.5f*(chroma(x-1, y, n) + chroma(x+1, y, n)) + green(x, y, n);

    Func v_interp("v_interp");
    v_interp(x, y, n) = 0.5f*(chroma(x, y-1, n) + chroma(x, y+1, n)) + green(x, y, n);

    Func q_interp("q_interp");  // quincux interpolation, when we have 4 neighbors
    q_interp(x, y, n) = 0.25f*(
        chroma(x-1, y-1, n) + 
        chroma(x-1, y+1, n) +
        chroma(x+1, y-1, n) +
        chroma(x+1, y+1, n)) + green(x, y, n);

    Func red("red");
    red(x, y, n) = select(is_g0, h_interp(x, y, n),
                       is_g3, v_interp(x, y, n),
                       is_blue, q_interp(x, y, n),
                       f_mosaick(x, y, n));

    Func blue("blue");
    blue(x, y, n) = select(is_g0, v_interp(x, y, n),
                        is_g3, h_interp(x, y, n),
                        is_red, q_interp(x, y, n),
                        f_mosaick(x, y, n));

    Func f_output("f_output");
    f_output(x, y, c, n) = select(c == 0, red(x, y, n),
                               c == 1, green(x, y, n),
                               blue(x, y, n));

    std::map<std::string, Func> func_map;
    func_map["mosaick"]  = f_mosaick;
    func_map["output"]  = f_output;

    return func_map;
}

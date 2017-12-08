#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> learnable_demosaick(
        const Input &mosaick,
        const Input &gfilt,
        const Input &grad_filt
        ) {
    Func f_mosaick("f_mosaick");
    f_mosaick(x, y, n) = Halide::BoundaryConditions::repeat_edge(
        mosaick)(x, y, n);
    Func f_gfilt("f_gfilt");
    Func f_grad_filt("f_grad_filt");
    f_gfilt(x) = gfilt(x);
    f_grad_filt(x) = grad_filt(x);

    Expr gfilt_sz = gfilt.dim(0).extent();
    Expr grad_filt_sz = gfilt.dim(0).extent();

    Expr is_green = (x % 2 == y % 2);
    Expr is_g0 = (x % 2 == 0) && (y % 2 == 0);
    Expr is_g3 = (x % 2 == 1) && (y % 2 == 1);
    Expr is_red = (x % 2 == 1) && (y % 2 == 0);
    Expr is_blue = (x % 2 == 0) && (y % 2 == 1);

    Func dx("dx");
    Func dy("dy");
    RDom rgrad(0, grad_filt_sz);
    dx(x, y, n) += 0.0f;
    dx(x, y, n) += f_mosaick(x + rgrad - grad_filt_sz/2, y, n)*f_grad_filt(rgrad);
    dx(x, y, n) = abs(dx(x, y, n));
    dy(x, y, n) += 0.0f;
    dy(x, y, n) += f_mosaick(x, y + rgrad - gfilt_sz/2, n)*f_grad_filt(rgrad);
    dy(x, y, n) = abs(dy(x, y, n));

    Func h_interp_g("h_interp_g");
    Func v_interp_g("v_interp_g");
    RDom r(0, gfilt_sz);
    h_interp_g(x, y, n) += 0.0f;
    h_interp_g(x, y, n) += f_mosaick(x + r - gfilt_sz/2, y, n)*f_gfilt(r);
    v_interp_g(x, y, n) += 0.0f;
    v_interp_g(x, y, n) += f_mosaick(x, y + r - gfilt_sz/2, n)*f_gfilt(r);

    float scale = 10.f;
    Expr mask = clamp(sigmoid((dx(x, y, n)-dy(x, y, n))*scale), 0.0f, 1.0f);

    Func interpolated_green("interpolated_green");
    interpolated_green(x, y, n) = v_interp_g(x, y, n)*mask + h_interp_g(x, y, n)*(1.0f-mask);

    Func green("green");
    green(x, y, n) = select(is_green, f_mosaick(x, y, n), interpolated_green(x, y, n));

    // -- Chroma interpolation ------------------------------------------------
    Func chroma("chroma");
    chroma(x, y, n) = f_mosaick(x, y, n) - green(x, y, n);

    Func h_interp("h_interp");
    h_interp(x, y, n) = 0.5f*(chroma(x-1, y, n) + chroma(x+1, y, n)) + green(x, y, n);

    Func v_interp("v_interp");
    v_interp(x, y, n) = 0.5f*(chroma(x, y-1, n) + chroma(x, y+1, n)) + green(x, y, n);

    // quincux interpolation, when we have 4 neighbors
    Func q_interp("q_interp");
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
    func_map["gfilt"]  = f_gfilt;
    func_map["grad_filt"]  = f_grad_filt;
    func_map["dx"]  = dx;
    func_map["dy"]  = dy;
    func_map["v_interp_g"]  = v_interp_g;
    func_map["h_interp_g"]  = h_interp_g;
    func_map["output"]  = f_output;

    return func_map;
}

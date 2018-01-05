#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n"), k("k");

template <typename Input>
std::map<std::string, Func> learnable_demosaick(
        const Input &mosaick,
        const Input &selection_filters,
        const Input &green_filters,
        const Input &h_chroma_filter,
        const Input &v_chroma_filter,
        const Input &q_chroma_filter
        ) {
    Func f_mosaick("f_mosaick");
    f_mosaick(x, y, n) = Halide::BoundaryConditions::repeat_edge(
        mosaick)(x, y, n);
    Func f_sel_filts("f_sel_filts");
    f_sel_filts(x, y, k) = selection_filters(x, y, k);
    Func f_green_filts("f_green_filts");
    f_green_filts(x, y, k) = green_filters(x, y, k);

    Func f_h_chroma_filter("f_h_chroma_filter");
    f_h_chroma_filter(x, y) = h_chroma_filter(x, y);

    Func f_v_chroma_filter("f_v_chroma_filter");
    f_v_chroma_filter(x, y) = v_chroma_filter(x, y);

    Func f_q_chroma_filter("f_q_chroma_filter");
    f_q_chroma_filter(x, y) = q_chroma_filter(x, y);

    Expr is_green = (x % 2 == y % 2);
    Expr is_g0 = (x % 2 == 0) && (y % 2 == 0);
    Expr is_g3 = (x % 2 == 1) && (y % 2 == 1);
    Expr is_red = (x % 2 == 1) && (y % 2 == 0);
    Expr is_blue = (x % 2 == 0) && (y % 2 == 1);

    Expr sel_filt_w = selection_filters.dim(0).extent();
    Expr sel_filt_h = selection_filters.dim(1).extent();
    Expr green_filt_w = green_filters.dim(0).extent();
    Expr green_filt_h = green_filters.dim(1).extent();

    Expr nfilters = green_filters.dim(2).extent();
    RDom rfilters(0, nfilters);

    Func selection("selection");
    RDom rsel(0, sel_filt_w, 0, sel_filt_h);
    selection(x, y, k, n) = 0.0f;
    selection(x, y, k, n) += f_mosaick(x + rsel.x - sel_filt_w/2, y + rsel.y - sel_filt_h/2, n)*f_sel_filts(rsel.x, rsel.y, k);
    Func abs_selection("abs_selection");
    abs_selection(x, y, k, n) = abs(selection(x, y, k, n));

    // Softmax ----------------
    Func sel_max("sel_max");
    sel_max(x, y, n) = 0.0f;
    sel_max(x, y, n) = max(sel_max(x, y,n), abs_selection(x, y, rfilters, n));
    Func exp_selection("exp_selection");
    // stable softmax (-= max)
    exp_selection(x, y, k, n) = exp((sel_max(x, y, n) - abs_selection(x, y, k, n)));
    Func normalizer("normalizer");
    normalizer(x, y, n) = 0.0f;
    normalizer(x, y, n) += exp_selection(x, y, rfilters, n);

    Func weights("weights");
    weights(x, y, k, n) = exp_selection(x, y, k, n) / normalizer(x, y, n);
    // ------------------------
    
    // Hard max -------------------
    // Func weights("weights");
    // weights(x, y, k, n) = 0.0f;
    // weights(x, y, 0, n) = select(abs_selection(x, y, 0, n) > abs_selection(x, y, 1, n), 0.0f, 1.0f);
    // weights(x, y, 1, n) = select(abs_selection(x, y, 0, n) > abs_selection(x, y, 1, n), 1.0f, 0.0f);
    // ------------------------

    Func interp_g("interp_g");
    RDom r(0, green_filt_w, 0, green_filt_h);
    interp_g(x, y, k, n) = 0.0f;
    interp_g(x, y, k, n) += f_mosaick(x + r.x - green_filt_w/2, y + r.y - green_filt_h/2, n)*f_green_filts(r.x, r.y, k);

    Func interpolated_green("interpolated_green");
    interpolated_green(x, y, n) = 0.0f;
    interpolated_green(x, y, n) += interp_g(x, y, rfilters, n)*weights(x, y, rfilters, n);

    Func green("green");
    green(x, y, n) = select(is_green, f_mosaick(x, y, n), interpolated_green(x, y, n));

    // -- Chroma interpolation ------------------------------------------------
    Func chroma("chroma");
    chroma(x, y, n) = f_mosaick(x, y, n) - green(x, y, n);

    Func h_interp("h_interp");
    Expr h_filt_w = h_chroma_filter.dim(0).extent();
    Expr h_filt_h = h_chroma_filter.dim(1).extent();
    RDom r_h(0, h_filt_w, 0, h_filt_h);
    h_interp(x, y, n) = 0.0f;
    h_interp(x, y, n) += chroma(x + r_h.x - h_filt_w/2, y + r_h.y - h_filt_h/2, n)
      * f_h_chroma_filter(r_h.x, r_h.y); // NOTE: only one filter for now
    // h_interp(x, y, n) = 0.5f*(chroma(x-1, y, n) + chroma(x+1, y, n)) + green(x, y, n);

    Func v_interp("v_interp");
    Expr v_filt_w = v_chroma_filter.dim(0).extent();
    Expr v_filt_h = v_chroma_filter.dim(1).extent();
    RDom r_v(0, v_filt_w, 0, v_filt_h);
    v_interp(x, y, n) = 0.0f;
    v_interp(x, y, n) += chroma(x + r_v.x - v_filt_w/2, y + r_v.y - v_filt_h/2, n)
      * f_v_chroma_filter(r_v.x, r_v.y); // NOTE: only one filter for now
    // v_interp(x, y, n) = 0.5f*(chroma(x, y-1, n) + chroma(x, y+1, n)) + green(x, y, n);

    // quincux interpolation, when we have 4 neighbors
    Func q_interp("q_interp");
    Expr q_filt_w = q_chroma_filter.dim(0).extent();
    Expr q_filt_h = q_chroma_filter.dim(1).extent();
    RDom r_q(0, q_filt_w, 0, q_filt_h);
    q_interp(x, y, n) = 0.0f;
    q_interp(x, y, n) += chroma(x + r_q.x - v_filt_w/2, y + r_q.y - v_filt_h/2, n)
      * f_q_chroma_filter(r_q.x, r_q.y); // NOTE: only one filter for now
    // q_interp(x, y, n) = 0.25f*(
    //     chroma(x-1, y-1, n) + 
    //     chroma(x-1, y+1, n) +
    //     chroma(x+1, y-1, n) +
    //     chroma(x+1, y+1, n)) + green(x, y, n);

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
    func_map["selection_filters"]  = f_sel_filts;
    func_map["green_filters"]  = f_green_filts;
    func_map["selection"]  = selection;
    func_map["h_chroma_filter"]  = f_h_chroma_filter;
    func_map["v_chroma_filter"]  = f_v_chroma_filter;
    func_map["q_chroma_filter"]  = f_q_chroma_filter;
    // func_map["normalizer"]  = normalizer;
    // func_map["sel_max"]  = sel_max;
    // func_map["exp_selection"]  = exp_selection;
    func_map["weights"]  = weights;
    func_map["interp_g"]  = interp_g;
    func_map["interpolated_green"]  = interpolated_green;
    func_map["output"]  = f_output;

    return func_map;
}

#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n"), k("k");

template <typename Input>
std::map<std::string, Func> learnable_demosaick(
        const Input &mosaick) {

    Func cfa("cfa");
    cfa(x, y, n) = Halide::BoundaryConditions::repeat_edge(
        mosaick)(x, y, n);

    Expr is_green = (x % 2 == y % 2);
    Expr is_g0 = (x % 2 == 0) && (y % 2 == 0);
    Expr is_g3 = (x % 2 == 1) && (y % 2 == 1);
    Expr is_red = (x % 2 == 1) && (y % 2 == 0);
    Expr is_blue = (x % 2 == 0) && (y % 2 == 1);

    float ar_thresh = 0.75;
    float eps = 1e-5;

    Func h_grad("h_grad");
    h_grad(x, y, n) = abs(cfa(x+1, y, n) - cfa(x-1, y, n));

    Func v_grad("v_grad");
    v_grad(x, y, n) = abs(cfa(x, y+1, n) - cfa(x, y-1, n));

    RDom rdwts(-1, 3, "rdwts");
    Func dir_wts_0("dir_wts_0");
    // NOTE: We can parametrize this
    dir_wts_0(x, y, n) = eps;
    dir_wts_0(x, y, n) += v_grad(x, y+rdwts, n);

    Func dir_wts_1("dir_wts_1");
    dir_wts_1(x, y, n) = eps;
    dir_wts_1(x, y, n) += h_grad(x+rdwts, y, n);

    Func delhvsqsum("delhvsqsum");
    delhvsqsum(x, y, n) = 
      h_grad(x, y, n)*h_grad(x, y, n) + v_grad(x, y, n)*v_grad(x, y, n);

    // ---------------------
    
    // interpolate with Hamilton Adams:
    Func g_ha("g_ha");
    // TODO the directional interpolation weight can be relaxed
    g_ha(x, y, n, k) = 0.0f; // k = u, d, l, r
    g_ha(x, y, n, 0) = cfa(x, y-1, n) + 0.5f*(cfa(x, y, n) - cfa(x, y-2, n));
    g_ha(x, y, n, 1) = cfa(x, y+1, n) + 0.5f*(cfa(x, y, n) - cfa(x, y+2, n));
    g_ha(x, y, n, 2) = cfa(x-1, y, n) + 0.5f*(cfa(x, y, n) - cfa(x-2, y, n));
    g_ha(x, y, n, 3) = cfa(x+1, y, n) + 0.5f*(cfa(x, y, n) - cfa(x+2, y, n));

    Func cr("cr");
    // TODO the ratio interpolation weight can be relaxed
    cr(x, y, n, k) = 0.0f; // k = u, d, l, r
    cr(x, y, n, 0) = 
      cfa(x, y-1, n) * (dir_wts_0(x, y-2, n)+dir_wts_0(x, y, n))
      / ((cfa(x, y, n) + eps)*dir_wts_0(x, y-2, n) + (cfa(x, y-2, n) + eps)*dir_wts_0(x, y, n));
    cr(x, y, n, 1) = 
      cfa(x, y+1, n) * (dir_wts_0(x, y+2, n)+dir_wts_0(x, y, n))
      / ((cfa(x, y, n) + eps)*dir_wts_0(x, y+2, n) + (cfa(x, y+2, n) + eps)*dir_wts_0(x, y, n));
    cr(x, y, n, 2) = 
      cfa(x-1, y, n) * (dir_wts_1(x-2, y, n)+dir_wts_1(x, y, n))
      / ((cfa(x, y, n) + eps)*dir_wts_1(x-2, y, n) + (cfa(x-2, y, n) + eps)*dir_wts_1(x, y, n));
    cr(x, y, n, 3) = 
      cfa(x+1, y, n) * (dir_wts_1(x+2, y, n)+dir_wts_1(x, y, n))
      / ((cfa(x, y, n) + eps)*dir_wts_1(x+2, y, n) + (cfa(x+2, y, n) + eps)*dir_wts_1(x, y, n));

    // interpolate with color ratios
    Func g_ar("g_u_ar");
    g_ar(x, y, n, k) = select(
        abs(1.0f - cr(x, y, n, k)) < ar_thresh,
        cfa(x, y, n)*cr_u(x, y, n, k), // ratio is close to one
        g_ha(x, y, n, k));

    Func hwt("hwt");
    hwt(x, y, n) = dir_wts_1(x-1, y, n) / (dir_wts_1(x-1, y, n) + dir_wts_1(x+1, y, n));
    Func vwt("vwt");
    vwt(x, y, n) = dir_wts_0(x, y-1, n) / (dir_wts_0(x, y-1, n) + dir_wts_0(x, y+1, n));

    // Interpolate G ha
    Func gint_v_ha("gint_v_ha");
    gint_v_ha(x, y, n) = vwt(x, y, n)*g_ha(x, y, n, 1) + (1-vwt(x, y, m))*g_ha(x, y, n, 0);
    Func gint_h_ha("gint_h_ha");
    gint_h_ha(x, y, n) = hwt(x, y, n)*g_ha(x, y, n, 3) + (1-hwt(x, y, m))*g_ha(x, y, n, 2);

    // Diagonal chroma gradients
    chroma_grad_se(x, y, k, n) = abs(chroma(x+1, y+1, k, n) - chroma(x-1, y-1, k, n));
    chroma_grad_ne(x, y, k, n) = abs(chroma(x+1, y-1, k, n) - chroma(x-1, y+1, k, n));

    // Weights for 4 cardinal direction, keep the symmetry
    RDom r(0, 2, 0, 2);

    // Interpolate chroma a R/B locations
    RDom rc(0, 4);
    chroma_rb(x, y, k, n) = weight(rc) * chroma(x, y, k, n);
    
    // Interpolate chroma a G locations
    chroma_g(x, y, k, n) = 
    
    // Add chroma + green to get final color

    return func_map;
}
//
    // TODO: We can parametrize this
    dir_wts_0(x, y, n) = eps;
    dir_wts_0(x, y, n) += v_grad(x, y+rdwts, n);

    Func dir_wts_1("dir_wts_1");
    dir_wts_1(x, y, n) = eps;
    dir_wts_1(x, y, n) += h_grad(x+rdwts, y, n);

    Func hwt("hwt");
    hwt(x, y, n) = dir_wts_1(x-1, y, n) / (dir_wts_1(x-1, y, n) + dir_wts_1(x+1, y, n));
    Func vwt("vwt");
    vwt(x, y, n) = dir_wts_0(x, y-1, n) / (dir_wts_0(x, y-1, n) + dir_wts_0(x, y+1, n));

    // ---------------------
    
    // // interpolate with Hamilton Adams:
    Func g_ha("g_ha");
    // TODO the directional interpolation weight can be relaxed
    g_ha(x, y, n, k) = 0.0f; // k = u, d, l, r
    g_ha(x, y, n, 0) = cfa(x, y-1, n) + 0.5f*(cfa(x, y, n) - cfa(x, y-2, n));
    g_ha(x, y, n, 1) = cfa(x, y+1, n) + 0.5f*(cfa(x, y, n) - cfa(x, y+2, n));
    g_ha(x, y, n, 2) = cfa(x-1, y, n) + 0.5f*(cfa(x, y, n) - cfa(x-2, y, n));
    g_ha(x, y, n, 3) = cfa(x+1, y, n) + 0.5f*(cfa(x, y, n) - cfa(x+2, y, n));

    // Interpolate G ha
    Func gint_v_ha("gint_v_ha");
    gint_v_ha(x, y, n) = vwt(x, y, n)*g_ha(x, y, n, 1) + (1-vwt)(x, y, m)*g_ha(x, y, n, 0);
    Func gint_h_ha("gint_h_ha");
    gint_h_ha(x, y, n) = hwt(x, y, n)*g_ha(x, y, n, 3) + (1-hwt(x, y, m))*g_ha(x, y, n, 2);

    // Compute h, v, diag gradients
    // Compute weights for up vs down using vertical gradients (3taps separable)
    // Compute weights for left vs right using vertical gradients (3taps separable)

    // interpolate green with HA, in 4 dir
    // interpolate green CR, if CR > thresh replace by HA, in 4 dir
    // h, v interpolation of green: weight depends on gradient before and after.
    //  if gradient before is small => pick before
    // CR is the primary interpolation, but keep HA as alternative
    //  (opt) replace primary by ha, if highlights almost clipped
    // compute square diff of interp in h, v dir (dgintv)
    //
    // Compute h, v color diff G - R/B for primary and alt
    // Compute h, v, variance of cd over [-2, 0, 2] for primary and alt
    //  if var(alt) < var(cd) replace cd by cd_alt
    //  (opt) bound interpolation in high sat areas
    // Compute square of h, v color diff and square diff btw h and v
    // Compute color diff variance in 4 directions over [0 -1 -2 -3]
    // Compute h, v cd variance (using dirwts gradient weight, hwt, vwt)
    // Compute fluctuation in u/d l/r color interp (dgintv) over [0 -1 -2]
    //   Compute cd variance h, v from these flucuation (using vwt, hwt)
    // Determine adaptive weight for G interpolation 
    //    (hcdvar/(vcdvar+hcdvar) for color difference varwt
    //    (hcdvar1/(vcdvar1+hcdvar1) for h/v interp diffwt
    // Determine the h,v interp weight between these two, if they agree on direction, pick
      // strongest discrimination. default to diff otherwise. hvwt
    // (ignore) Nyquist test
    // (ignore) compute area interpolation weight for nyquist test
    // (ignore) do area interpolation in nyquist pix, updates hvwt
    // Populate G at R/B
    
    // Color difference
    Func h_cd("h_cd");
    h_cd(x, y, n) = select(is_green, 1.0f, -1.0f)*(gint_h_ha(x, y, n) - cfa)

    Func s_chroma("s_chroma");  // sparse chroma red in chan 0, blue in chan 1
    s_chroma(x, y, n) = r_green(x, y, n) - cfa(x, y, n);

    // Diagonal chroma gradients
    Func chroma_grad_se("chroma_grad_se");
    Func chroma_grad_ne("chroma_grad_ne");
    chroma_grad_se(x, y, n) = abs(s_chroma(x+1, y+1, n) - s_chroma(x-1, y-1, n));
    chroma_grad_ne(x, y, n) = abs(s_chroma(x+1, y-1, n) - s_chroma(x-1, y+1, n));

    // Weights for 4 cardinal direction, keep the symmetry
    RDom r(0, 3);
    Func weight("weight");
    // weight(x, y, n, dx, dy) = 

    // Interpolate chroma a R/B locations
    RDom rc(0, 2, 0, 2);
    chroma_rb(x, y, n) = weight(x, y, n, rc.x, rc.y) * chroma(x - 1 + 2*rc.x, y - 1 + 2*rc.y, n);
    
    // Interpolate chroma a G locations
    chroma_g(x, y, k, n) = 
    
    // Add chroma + green to get final color
    //
    // Func cr("cr");
    // // TODO the ratio interpolation weight can be relaxed
    // cr(x, y, n, k) = 0.0f; // k = u, d, l, r
    // cr(x, y, n, 0) = 
    //   cfa(x, y-1, n) * (dir_wts_0(x, y-2, n)+dir_wts_0(x, y, n))
    //   / ((cfa(x, y, n) + eps)*dir_wts_0(x, y-2, n) + (cfa(x, y-2, n) + eps)*dir_wts_0(x, y, n));
    // cr(x, y, n, 1) = 
    //   cfa(x, y+1, n) * (dir_wts_0(x, y+2, n)+dir_wts_0(x, y, n))
    //   / ((cfa(x, y, n) + eps)*dir_wts_0(x, y+2, n) + (cfa(x, y+2, n) + eps)*dir_wts_0(x, y, n));
    // cr(x, y, n, 2) = 
    //   cfa(x-1, y, n) * (dir_wts_1(x-2, y, n)+dir_wts_1(x, y, n))
    //   / ((cfa(x, y, n) + eps)*dir_wts_1(x-2, y, n) + (cfa(x-2, y, n) + eps)*dir_wts_1(x, y, n));
    // cr(x, y, n, 3) = 
    //   cfa(x+1, y, n) * (dir_wts_1(x+2, y, n)+dir_wts_1(x, y, n))
    //   / ((cfa(x, y, n) + eps)*dir_wts_1(x+2, y, n) + (cfa(x+2, y, n) + eps)*dir_wts_1(x, y, n));
    //
    // // interpolate with color ratios
    // Func g_ar("g_u_ar");
    // g_ar(x, y, n, k) = select(
    //     abs(1.0f - cr(x, y, n, k)) < ar_thresh,
    //     cfa(x, y, n)*cr_u(x, y, n, k), // ratio is close to one
    //     g_ha(x, y, n, k));

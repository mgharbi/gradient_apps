#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> burst_demosaicking(
        const Input &inputs,
        const Input &homographies,
        const Input &reconstructed
        ) {
    Func f_inputs("f_inputs");
    f_inputs(x, y, n) = inputs(x, y, n);  // grayscale mosaicks
    Func f_reconstructed("f_reconstructed");
    f_reconstructed(x, y, c) = 
      Halide::BoundaryConditions::constant_exterior(reconstructed, 0.0f)(x, y, c);
    Func f_homographies("f_homographies");
    f_homographies(x, n) = homographies(x, n);

    Expr width = inputs.dim(0).extent();
    Expr height = inputs.dim(1).extent();
    Expr count = inputs.dim(2).extent();

    // Forward warp
    Func warped("warped");
    warped(x, y, c, n) = 0.0f;
      //   reconstructed(fx,   fy,   c, n)*(1.0f-wx)*(1.0f-wy)
      // + reconstructed(fx,   fy+1, c, n)*(1.0f-wx)*(     wy)
      // + reconstructed(fx+1, fy,   c, n)*(     wx)*(1.0f-wy)
      // + reconstructed(fx+1, fy+1, c, n)*(     wx)*(     wy);
    
    // Re-mosaick
    // G R
    // B G
    Func mosaicked("mosaicked");
    mosaicked(x, y, n) = select(
        x % 2 == 1 && y % 2 == 0, warped(x, y, 0, n),  // red
        x % 2 == 0 && y % 2 == 1, warped(x, y, 2, n),  // blue
        warped(x, y, 2, n));  // green

    // Gradient prior term
    Func dx("dx");
    Func dy("dy");
    dx(x, y, c) = abs(reconstructed(x+1, y, c) - reconstructed(x, y, c));
    dy(x, y, c) = abs(reconstructed(x, y+1, c) - reconstructed(x, y, c));
    Func gradient_loss("gradient_loss");
    RDom r(0, width, 0, height, 0, 3);
    gradient_loss(x) = 0.0f;
    gradient_loss(x) += dx(r.x, r.y, r.z) + dy(r.x, r.y, r.z);

    // Data term
    RDom r2(0, width, 0, height, 0, count);
    Func data_loss("data_loss");
    Expr diff = mosaicked(r2.x, r2.y, r2.z) - f_inputs(r2.x, r2.y, r2.z);
    data_loss(x) = 0.0f;
    data_loss(x) += diff*diff;
    
    Func loss("loss");
    loss(x) = data_loss(x) + gradient_loss(x);

    std::map<std::string, Func> func_map;
    func_map["inputs"]  = inputs;
    func_map["homographies"]  = homographies;
    func_map["reconstructed"]  = f_reconstructed;
    func_map["loss"]  = loss;

    return func_map;
}

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
        // const Input &confidence,
        const Input &homographies,
        const Input &reconstructed,
        const Input &gradient_weight
        ) {
    Func f_inputs("f_inputs");
    f_inputs(x, y, n) = Halide::BoundaryConditions::constant_exterior(
        inputs, 0.0f)(x, y, n);
    // Func f_confidence("f_confidence");
    // f_confidence(x, y, n) = confidence(x, y, n);
    Func f_reconstructed("f_reconstructed");
    f_reconstructed(x, y, c) = Halide::BoundaryConditions::constant_exterior(
        reconstructed, 0.0f)(x, y, c);
    Func f_h("f_h");
    f_h(x, n) = homographies(x, n);
    Func f_gradient_weight("f_gradient_weight");
    f_gradient_weight(x) = gradient_weight(x);

    Expr width = inputs.dim(0).extent();
    Expr height = inputs.dim(1).extent();
    Expr count = inputs.dim(2).extent();
    Expr r_width = reconstructed.dim(0).extent();
    Expr r_height = reconstructed.dim(1).extent();

    // Normalize coordinates, simplies multiscale approach if we need it
    Expr nrm_x = (x * 1.0f / width);
    Expr nrm_y = (y * 1.0f / height);

    // Fix last coeff of the 3x3 homography matrix to 1.0f to remove scale
    // ambiguity.
    Expr denom = f_h(6, n)*nrm_x + f_h(7, n)*nrm_y + 1.0f;
    Expr xformed_x = (f_h(0, n)*nrm_x + f_h(1, n)*nrm_y + f_h(2, n)) / denom;
    Expr xformed_y = (f_h(3, n)*nrm_x + f_h(4, n)*nrm_y + f_h(5, n)) / denom;
    
    // Convert back to image space, clamp for bounds_inferencer, but still
    // allow out of image samples.
    Expr new_x = clamp(
        r_width*xformed_x, -1.0f, cast<float>(r_width));
    Expr new_y = clamp(
        r_height*xformed_y, -1.0f, cast<float>(r_height));
    
    // Bilinear interpolation
    Expr fx = cast<int>(floor(new_x));
    Expr fy = cast<int>(floor(new_y));
    Expr wx = (new_x - fx);
    Expr wy = (new_y - fy);

    // Forward warp
    Func warped("warped");
    warped(x, y, c, n) = 
        f_reconstructed(fx,   fy,   c)*(1.0f-wx)*(1.0f-wy)
      + f_reconstructed(fx,   fy+1, c)*(1.0f-wx)*(     wy)
      + f_reconstructed(fx+1, fy,   c)*(     wx)*(1.0f-wy)
      + f_reconstructed(fx+1, fy+1, c)*(     wx)*(     wy);

    // Mask when the reconstruction reprojects outside the data
    Func mask("mask");
    mask(x, y, n) = select(
        new_x > 1.0f && new_x < cast<float>(r_width-1) &&
        new_y > 1.0f && new_y < cast<float>(r_height-1),
        1.0f, 0.0f);
    
    // Re-mosaick
    // G R
    // B G
    Func remosaicked("remosaicked");
    remosaicked(x, y, n) = select(
        x % 2 == 1 && y % 2 == 0, warped(x, y, 0, n),  // red
        x % 2 == 0 && y % 2 == 1, warped(x, y, 2, n),  // blue
        warped(x, y, 1, n));  // green

    // Data term
    RDom r2(0, width, 0, height, 0, count);
    Func reproj_error("reproj_error");
    reproj_error(x, y, n) = 
      (remosaicked(x, y, n) - f_inputs(x, y, n))*mask(x, y, n);
    Func data_loss("data_loss");
    Expr diff = reproj_error(r2.x, r2.y, r2.z);
    // Expr eps = 1e-6f;
    // data_loss(x) += sqrt(diff*diff + eps);
    // data_loss(x) += sqrt(diff*diff + eps)*exp(-f_confidence(r2.x, r2.y, r2.z));
    data_loss(x) = 0.0f;
    data_loss(x) += 0.5f*diff*diff;
    
    // Gradient prior term
    Func dx("dx");
    Func dy("dy");
    // Func luma("luma");
    // luma(x, y) = 
    //   0.25f*f_reconstructed(x, y, 0) +
    //   0.5f*f_reconstructed(x, y, 1) +
    //   0.25f*f_reconstructed(x, y, 2);
    // dx(x, y) = abs(luma(x+1, y) - luma(x, y));
    // dy(x, y) = abs(luma(x, y+1) - luma(x, y));

    dx(x, y, c) = abs(f_reconstructed(x+1, y, c) - f_reconstructed(x, y, c));
    dy(x, y, c) = abs(f_reconstructed(x, y+1, c) - f_reconstructed(x, y, c));
    Func gradient_loss("gradient_loss");
    RDom r(0, width, 0, height, 0, 3);
    gradient_loss(x) = 0.0f;
    gradient_loss(x) += dx(r.x, r.y, r.z) + dy(r.x, r.y, r.z);

    // Func gray_reproj("gray_reproj");
    // gray_reproj(x, y, n) = warped(x, y, 0, n);
    //
    // Func confidence_loss("confidence_loss");
    // confidence_loss(x) = 0.0f;
    // confidence_loss(x) += f_confidence(r2.x, r2.y, r2.z);
    
    Func loss("loss");
    loss(x) = 
      data_loss(0)/(1.0f*height*width*count) +
      f_gradient_weight(0)*gradient_loss(0)/(1.0f*height*width*3.0f);
      // confidence_loss(x)/(1.0f*height*width*count);

    std::map<std::string, Func> func_map;
    func_map["inputs"]  = f_inputs;
    // func_map["confidence"]  = f_confidence;
    func_map["homographies"] = f_h;
    func_map["reconstructed"] = f_reconstructed;
    func_map["reproj"] = reproj_error ;
    func_map["loss"]  = loss;

    return func_map;
}

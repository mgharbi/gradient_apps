#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "fft.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> learnable_wiener(
        const Input &blurred,
        const Input &kernel,
        const Input &reg_kernel
        ) {
    int W = blurred.width();
    int H = blurred.height();
    Func kernel_ce = BoundaryConditions::constant_exterior(kernel, 0.f);
    Func kernel_padded("kernel_padded");
    Expr u = select(x < (W - x), x, x - W) + kernel.width()  / 2;
    Expr v = select(y < (H - x), y, y - H) + kernel.height() / 2;
    kernel_padded(x, y) = kernel_ce(u, v);

    Fft2dDesc fwd_desc;
    Fft2dDesc inv_desc;
    inv_desc.gain = 1.0f / (W*H);

    // Compute the DFT of the input and the kernel.
    Func blurred_r("blurred_r");
    blurred_r(x, y) = blurred(x, y, 0);
    Func blurred_g("blurred_g");
    blurred_g(x, y) = blurred(x, y, 1);
    Func blurred_b("blurred_b");
    blurred_b(x, y) = blurred(x, y, 2);
    Target target = get_target_from_environment();
    ComplexFunc dft_in_r = fft2d_r2c(blurred_r, W, H, target, fwd_desc);
    ComplexFunc dft_in_g = fft2d_r2c(blurred_g, W, H, target, fwd_desc);
    ComplexFunc dft_in_b = fft2d_r2c(blurred_b, W, H, target, fwd_desc);
    ComplexFunc dft_kernel = fft2d_r2c(kernel_padded, W, H, target, fwd_desc);

    Func reg_kernel_func;
    reg_kernel_func(x, y) = reg_kernel(x, y);
    // Compute Wiener filter.
    ComplexFunc filter("wiener_filter");
    filter(x, y) = conj(dft_kernel(x, y) * 
        (re(dft_kernel(x, y) * conj(dft_kernel(x, y))) +
         reg_kernel_func(x, y) * reg_kernel_func(x, y) + 1e-6f));
    ComplexFunc dft_wiener_r("dft_wiener_r");
    ComplexFunc dft_wiener_g("dft_wiener_g");
    ComplexFunc dft_wiener_b("dft_wiener_b");
    dft_wiener_r(x, y) = dft_in_r(x, y) * filter(x, y);
    dft_wiener_g(x, y) = dft_in_g(x, y) * filter(x, y);
    dft_wiener_b(x, y) = dft_in_b(x, y) * filter(x, y);

    // Compute the inverse DFT to get the result.
    Func filtered_r = fft2d_c2r(dft_wiener_r, W, H, target, inv_desc);
    Func filtered_g = fft2d_c2r(dft_wiener_g, W, H, target, inv_desc);
    Func filtered_b = fft2d_c2r(dft_wiener_b, W, H, target, inv_desc);

    Func filtered("filtered");
    filtered(x, y, c) = 0.f;
    filtered(x, y, 0) = filtered_r(x, y);
    filtered(x, y, 1) = filtered_g(x, y);
    filtered(x, y, 2) = filtered_b(x, y);

    std::map<std::string, Func> func_map;
    func_map["filtered"] = filtered;
    func_map["reg_kernel"] = reg_kernel_func;

    return func_map;
}

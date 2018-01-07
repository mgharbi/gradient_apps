#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), c("c"), cf("cf"), n("n");

template <typename InputBuf, typename InputFloat, typename InputInt>
std::map<std::string, Func> non_local_means(
        const InputBuf   &input,
        const InputBuf   &feature_filter,
        const InputBuf   &patch_filter,
        const InputFloat &inv_sigma,
        const InputInt   &search_area) {
    Func clamped("clamped");
    clamped(x, y, c, n) =
      Halide::BoundaryConditions::repeat_edge(input)(x, y, c, n);
    Func feature_filter_func("feature_filter_func");
    feature_filter_func(x, y, c, cf) = feature_filter(x, y, c, cf);
    Func patch_filter_func("patch_filter_func");
    patch_filter_func(x, y) = patch_filter(x, y);
    Func inv_sigma_func("inv_sigma_func");
    inv_sigma_func() = inv_sigma;

    // Convolve the image with filters to obtain feature maps
    RDom r_ff(feature_filter);
    Func feature("feature");
    feature(x, y, cf, n) = 0.f;
    feature(x, y, cf, n) += clamped(x + r_ff.x - feature_filter.width()  / 2,
                                    y + r_ff.y - feature_filter.height() / 2,
                                    r_ff.z,
                                    n) * feature_filter_func(r_ff.x, r_ff.y, r_ff.z, cf);

    Var dx("dx"), dy("dy");
    Func dc("d");
    dc(x, y, dx, dy, cf, n) = pow(feature(x, y, cf, n) - feature(x + dx, y + dy, cf, n), 2);

    // Sum across color channels
    RDom channels(0, feature_filter.dim(3).extent());
    Func d("d");
    d(x, y, dx, dy, n) = sum(dc(x, y, dx, dy, channels, n));

    // Find the patch differences by blurring the difference images
    RDom patch_dom(patch_filter);
    Func blur_d("blur_d");
    blur_d(x, y, dx, dy, n) = 0.f;
    blur_d(x, y, dx, dy, n) += d(x + patch_dom.x - patch_filter.width()  / 2,
                                 y + patch_dom.y - patch_filter.height() / 2,
                                 dx,
                                 dy,
                                 n) * patch_filter_func(patch_dom.x, patch_dom.y);

    // Compute the weights from the patch differences
    Func w("w");
    Expr inv_sigma_sq = -(inv_sigma_func() * inv_sigma_func()) / (patch_filter.width() * patch_filter.height());
    w(x, y, dx, dy, n) = exp(blur_d(x, y, dx, dy, n) * inv_sigma_sq);

    // Add an alpha channel
    Func clamped_with_alpha("clamped_with_alpha");
    clamped_with_alpha(x, y, c, n) = select(c == 0, clamped(x, y, 0, n),
                                            c == 1, clamped(x, y, 1, n),
                                            c == 2, clamped(x, y, 2, n),
                                            1.0f);

    // Define a reduction domain for the search area
    RDom s_dom(-search_area/2, search_area, -search_area/2, search_area);

    // Compute the sum of the pixels in the search area
    Func non_local_means_sum("non_local_means_sum");
    non_local_means_sum(x, y, c, n) = 0.f;
    non_local_means_sum(x, y, c, n) += w(x, y, s_dom.x, s_dom.y, n) * clamped_with_alpha(x + s_dom.x, y + s_dom.y, c, n);

    Func output("output");
    output(x, y, c, n) =
        non_local_means_sum(x, y, c, n) / (non_local_means_sum(x, y, 3, n) + 1e-6f);

    std::map<std::string, Func> func_map;
    func_map["clamped"]  = clamped;
    func_map["feature_filter_func"]  = feature_filter_func;
    func_map["patch_filter_func"] = patch_filter_func;
    func_map["inv_sigma_func"] = inv_sigma_func;
    func_map["output"] = output;
    return func_map;
}

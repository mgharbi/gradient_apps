#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), n("n"), ci("ci"), co("co");

template <typename Input>
std::map<std::string, Func> bilateral_layer(
		const Input &input,
	    const Input &guide,
	    const Input &filter,
  	    const Input &bias) {
    // TODO: should we downsample the input here?

    // Offsets the input by different biases and applies ReLU
    // Do this for each channel
    Func f_input("input");
    f_input(x, y, ci, n) = input(x, y, ci, n);
    Func f_guide("guide");
    f_guide(x, y, n) = guide(x, y, n);
    Func f_bias("bias");
    f_bias(z, ci) = bias(z, ci);
    Func f_offset("offset");
    // x, y, z offset, input channel, batch size
    f_offset(x, y, z, ci, n) = max(0.f, f_input(x, y, ci, n) + f_bias(z, ci));
    // Perform 3D filtering in the offseted space
    // Again, do this for each channel
    // We assume the z offset part is fully-connected 
    // (i.e. filter.dim(2).extent() == bias.dim(1).extent())
    RDom r(filter.dim(0).min(), filter.dim(0).extent(),  // x
           filter.dim(1).min(), filter.dim(1).extent(),  // y
           filter.dim(2).min(), filter.dim(2).extent(),  // z offset
           filter.dim(3).min(), filter.dim(3).extent()); // input channel
    Func f_filter("filter");
    f_filter(x, y, z, ci, co) = filter(x, y, z, ci, co);
    Func f_conv("conv");
    f_conv(x, y, z, co, n)  = 0.f;
    f_conv(x, y, z, co, n) += f_filter(r[0], r[1], r[2], r[3], co) *
                              f_offset(x + r[0], y + r[1], r[2], r[3], n);
    // Slice the result back to 2D
    // Find the coordinate in z
    Expr gz = clamp(f_guide(x, y, n), 0.0f, 1.0f) * (filter.dim(2).extent() - 1);
    // Floor voxel
    Expr fz = cast<int>(floor(gz) - 0.5f);
    // Ceil voxel
    Expr cz = fz + 1;
    // Weight
    Expr wz = gz - fz;
    // Linear interpolation
    Func f_output("output");
    f_output(x, y, co, n) = f_conv(x, y, fz, co, n) * (1.f - wz) +
                            f_conv(x, y, cz, co, n) * wz;

    std::map<std::string, Func> func_map;
    func_map["input"]  = f_input;
    func_map["guide"]  = f_guide;
    func_map["bias"]   = f_bias;
    func_map["offset"] = f_offset;
    func_map["filter"] = f_filter;
    func_map["conv"]   = f_conv;
    func_map["output"] = f_output;

    return func_map;
}
#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), ci("ci"), co("co"), n("n");

Expr relu(Expr expr) {
  return max(expr, 0.0f);
}

void conv(const Func input, const Func weights, const Func biases, 
          const int ksize, const int chans, Func &conv, bool do_relu=true) {
  RDom r(0, ksize, 0, ksize, 0, chans);
  conv(x, y, co, n) = biases(co);
  if(do_relu) {
    conv(x, y, co, n) += relu(input(x+r.x - ksize/2, y+r.y - ksize/2, r.z, n))
      * weights(r.x, r.y, r.z, co);
  } else {
    conv(x, y, co, n) += input(x+r.x - ksize/2, y+r.y - ksize/2, r.z, n)
      * weights(r.x, r.y, r.z, co);
  }
}

void fc(const Func input, const Func weights, const Func biases, 
        const int chans, Func &out, bool do_relu=true) {
  RDom r(0, chans);
  out(co, n) = biases(co);
  if(do_relu) {
    out(co, n) += relu(input(r, n))*weights(r, co);
  } else {
    out(co, n) += input(r, n)*weights(r, co);
  }
}

void maxpool(const Func input, const int stride, Func &out) {
  RDom r(0, stride, 0, stride);
  out(x, y, c, n) = maximum(input(stride*x + r.x, stride*y + r.y, c, n));
}


template <typename Input, typename InputArray, typename InputArray2, typename InputArray3>
std::map<std::string, Func> vgg(
        const Input &input,
        const InputArray &weights,
        const InputArray2 &fc_weights,
        const InputArray3 &biases
        ) {
    Func f_input("f_input");
    f_input(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
        input, 0.0f)(x, y, ci, n);

    // vgg 16
    Func conv1_1("conv1_1"); conv(f_input, weights[0], biases[0], 3, 3, conv1_1, false);
    Func conv1_2("conv1_2"); conv(conv1_1, weights[1], biases[1], 3, 64, conv1_2);
    Func pool1("pool1"); maxpool(conv1_2, 2, pool1);

    Func conv2_1("conv2_1"); conv(pool1, weights[2], biases[2], 3, 64, conv2_1);
    Func conv2_2("conv2_2"); conv(conv2_1, weights[3], biases[3], 3, 128, conv2_2);
    Func pool2("pool2"); maxpool(conv2_2, 2, pool2);

    Func conv3_1("conv3_1"); conv(pool2, weights[4], biases[4], 3, 128, conv3_1);
    Func conv3_2("conv3_2"); conv(conv3_1, weights[5], biases[5], 3, 256, conv3_2);
    Func conv3_3("conv3_3"); conv(conv3_2, weights[6], biases[6], 3, 256, conv3_3);
    Func pool3("pool3"); maxpool(conv3_3, 2, pool3);

    Func conv4_1("conv4_1"); conv(pool3, weights[7],   biases[7], 3, 256, conv4_1);
    Func conv4_2("conv4_2"); conv(conv4_1, weights[8], biases[8], 3, 512, conv4_2);
    Func conv4_3("conv4_3"); conv(conv4_2, weights[9], biases[9], 3, 512, conv4_3);
    Func pool4("pool4"); maxpool(conv4_3, 2, pool4);

    Func conv5_1("conv5_1"); conv(pool4, weights[10],   biases[10], 3, 512, conv5_1);
    Func conv5_2("conv5_2"); conv(conv5_1, weights[11], biases[11], 3, 512, conv5_2);
    Func conv5_3("conv5_3"); conv(conv5_2, weights[12], biases[12], 3, 512, conv5_3);
    Func pool5("pool5"); maxpool(conv5_3, 2, pool5);

    Func vectorize("vectorize");
    Expr chan = co / (7*7);
    Expr yy = (co % (7*7)) / 7;
    Expr xx = co % 7;
    vectorize(co, n) = pool5(xx, yy, chan, n);

    Func fc6("fc6"); fc(vectorize, fc_weights[0], biases[13], 512, fc6);
    Func fc7("fc7"); fc(fc6, fc_weights[1], biases[14], 4096, fc7);
    Func fc8("fc8"); fc(fc7, fc_weights[2], biases[15], 4096, fc8);

    std::map<std::string, Func> func_map;
    func_map["input"] = f_input;
    func_map["output"] = fc8;

    return func_map;
}

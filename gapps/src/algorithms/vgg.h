#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), c("c"), ci("ci"), co("co"), n("n");

void conv(const Func input, const Func weights, const Func biases, 
          const int ksize, const int chans, Func &conv) {
  RDom r(0, ksize, 0, ksize, 0, chans);
  conv(x, y, co, n) = biases(co);
  conv(x, y, co, n) += input(x+r.x - ksize/2, y+r.y - ksize/2, r.z, n);
  return conv;
}


template <typename Input>
std::map<std::string, Func> vgg(
        const Input &input,
        const std::vector<Input> &weights,
        const std::vector<Input> &biases
        ) {
    Func f_input("f_input");
    f_input(x, y, n) = Halide::BoundaryConditions::constant_exterior(
        input, 0.0f)(x, y, n);

    // vgg 16
    Func conv1_1("conv1_1"); conv(f_inputs, weights[0], biases[0], 3, 3, conv1_1);
    Func conv1_2("conv1_2"); conv(conv1_1, weights[0], biases[0], 3, 64, conv1_2);

    Func output("output");
    output(x, y, c, n) = conv1_2(x, y, c, n);

    // conv 3 64
    // conv 3 64
    // maxpool
    
    // conv 3 128
    // conv 3 128
    // maxpool
    
    // conv 3 256
    // conv 3 256
    // conv 3 256
    // maxpool
    
    // conv 3 512
    // conv 3 512
    // conv 3 512
    // maxpool
    
    // conv 3 512
    // conv 3 512
    // conv 3 512
    // maxpool
    
    // fc 4096
    // fc 4096
    // fc 1000
    // softmax

    std::map<std::string, Func> func_map;
    func_map["inputs"]  = f_inputs;
    // func_map["confidence"]  = f_confidence;
    func_map["output"]  = output;

    return func_map;
}

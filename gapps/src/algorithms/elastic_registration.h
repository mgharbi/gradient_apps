#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y");

template <typename Input>
std::map<std::string, Func> elastic_registration(
        const Input &input,
        const Input &target) {
    
}

#pragma once

#include "Halide.h"

using Halide::Expr;

Expr sigmoid(Expr x) {
  return 1.0f / (1.0f*exp(-x));
}

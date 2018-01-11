#define INIT 0
#include "deconv_grad_backward.hl.cxx"

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvGradBackwardGenerator, deconv_grad_iter_backward)


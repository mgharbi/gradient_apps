#define INIT 1
#include "deconv_grad_backward.hl.cxx"

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvGradBackwardGenerator, deconv_grad_init_backward)


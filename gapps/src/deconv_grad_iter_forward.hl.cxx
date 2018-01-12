#define INIT 0
#include "deconv_grad_forward.hl.cxx"

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvGradForwardGenerator, deconv_grad_iter_forward)


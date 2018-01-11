#define INIT 1
#include "deconv_grad_forward.hl.cxx"

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvGradForwardGenerator, deconv_grad_init_forward)


#define NEED_HESS
#include "deconv_grad_backward.h"

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvGradBackwardGenerator, deconv_grad_hess_backward)

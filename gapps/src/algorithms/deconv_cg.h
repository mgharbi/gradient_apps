#pragma once

#include <map>
#include <string>
#include <Halide.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), n("n");

struct Input {
    template <typename Buf>
    Input(const Buf &buffer) {
        func = buffer;
        for (int d = 0; d < buffer.dimensions(); d++) {
            int min = buffer.dim(d).min();
            int extent = buffer.dim(d).extent();
            bounds.push_back(std::make_pair(min, extent));
        }
    }

    int min(int d) const {
        return bounds[d].first;
    }

    int extent(int d) const {
        return bounds[d].second;
    }

    Func func;
    std::vector<std::pair<int, int>> bounds;
};

Func deconv_cg_init(
        const Input &blurred,
        const Input &x0,
        const Input &kernel,
        const Input &reg_kernel_weights,
        const Input &reg_kernels,
        const Input &reg_targets,
        const Input &precond_kernel,
        const Input &w_kernel,
        const Input &w_reg_kernels) {
    // Initializing conjugate gradient
    // Want to solve A^TAx = A^Tb
    // A -> correlation with kernel
    // A^T -> convolution with kernel
    // Initializing r0 = A^Tb - A^TAx0
    RDom r_kernel(kernel.min(0), kernel.extent(0),
                  kernel.min(1), kernel.extent(1));
    Func clamped_b = BoundaryConditions::repeat_edge(blurred,
                {{Expr(blurred.min(0)), Expr(blurred.extent(0))},
                 {Expr(blurred.min(1)), Expr(blurred.extent(1))},
                 {Expr(), Expr()}});
    Func clamped_x0 = BoundaryConditions::repeat_edge(x0_func,
                {{Expr(0), Expr(w)},
                 {Expr(0), Expr(h)},
                 {Expr(), Expr()}});
    Func clamped_rtarget = BoundaryConditions::repeat_edge(reg_targets_func,
                {{Expr(0), Expr(w)},
                 {Expr(0), Expr(h)},
                 {Expr(), Expr()},
                 {Expr(), Expr()}});
    Func clamped_w_kernel = BoundaryConditions::repeat_edge(w_kernel_func,
                {{Expr(0), Expr(w)},
                 {Expr(0), Expr(h)},
                 {Expr(), Expr()}});
    Func clamped_w_reg_kernels = BoundaryConditions::repeat_edge(w_reg_kernels_func,
                {{Expr(0), Expr(w)},
                 {Expr(0), Expr(h)},
                 {Expr(), Expr()},
                 {Expr(), Expr()}});

    Func wkb("wkb");
    wkb(x, y, c) = clamped_w_kernel(x, y, c) * clamped_b(x, y, c);
    Func KTWb("K^TWb");
    KTWb(x, y, c) = 0.f;
    KTWb(x, y, c) += wkb(x - r_kernel.x + kw / 2,
                         y - r_kernel.y + kh / 2,
                         c) *
                     kernel(r_kernel.x, r_kernel.y);
    RDom r_reg_kernel_xy(0, rkw, 0, rkh);
    RDom r_reg_kernel_z(0, rkn);
    Func wrkb("wrkb");
    wrkb(x, y, c, n) = clamped_w_reg_kernels(x, y, c, n) * clamped_rtarget(x, y, c, n);
    Func rKTWb("rK^TWb");
    rKTWb(x, y, c, n) = 0.f;
    rKTWb(x, y, c, n) += wrkb(x - r_reg_kernel_xy.x + rkw / 2,
                              y - r_reg_kernel_xy.y + rkh / 2,
                              c,
                              n) *
                         reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);

    Func ATWb("A^TWb");
    ATWb(x, y, c) = KTWb(x, y, c);
    ATWb(x, y, c) += rKTWb(x, y, c, r_reg_kernel_z);
    Func Kx0("Kx0");
    Kx0(x, y, c) = 0.f;
    Kx0(x, y, c) += clamped_x0(x + r_kernel.x - kw / 2,
                               y + r_kernel.y - kh / 2,
                               c) *
                    kernel(r_kernel.x, r_kernel.y);
    Func WKx0("WKx0");
    WKx0(x, y, c) = Kx0(x, y, c) * clamped_w_kernel(x, y, c);
    Func KTWKx0("K^TWKx0");
    KTWKx0(x, y, c)  = 0.f;
    KTWKx0(x, y, c) += WKx0(x - r_kernel.x + kw / 2,
                            y - r_kernel.y + kh / 2,
                            c) *
                        kernel(r_kernel.x, r_kernel.y);
    Func rKx0("rKx0");
    rKx0(x, y, c, n) = 0.f;
    rKx0(x, y, c, n) += clamped_x0(x + r_reg_kernel_xy.x - rkw / 2,
                                   y + r_reg_kernel_xy.y - rkh / 2,
                                   c) *
                        reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func WrKx0("WrKx0");
    WrKx0(x, y, c, n) = rKx0(x, y, c, n) * clamped_w_reg_kernels(x, y, c, n);
    Func rKTWrKx0("rK^TWrKx0");
    rKTWrKx0(x, y, c, n) = 0.f;
    rKTWrKx0(x, y, c, n) += WrKx0(x - r_reg_kernel_xy.x + rkw / 2,
                                  y - r_reg_kernel_xy.y + rkh / 2,
                                  c,
                                  n) *
                            reg_kernels_func(r_reg_kernel_xy.x,
                                             r_reg_kernel_xy.y,
                                             n);

    Func ATWAx0("A^TWAx0");
    ATWAx0(x, y, c) = KTWKx0(x, y, c);
    ATWAx0(x, y, c) += rKTWrKx0(x, y, c, r_reg_kernel_z.x) *
                       reg_kernel_weights_func(r_reg_kernel_z.x) *
                       reg_kernel_weights_func(r_reg_kernel_z.x);

    Func r0("r0");
    r0(x, y, c) = ATWb(x, y, c) - ATWAx0(x, y, c);
    Func clamped_r0 = BoundaryConditions::repeat_edge(r0,
                {{Expr(0), Expr(w)},
                 {Expr(0), Expr(h)},
                 {Expr(), Expr()}});
    RDom r_precond_kernel(precond_kernel);
    Func Pr0("Pr0");
    Pr0(x, y, c) = 0.f;
    Pr0(x, y, c) += clamped_r0(x + r_precond_kernel.x - precond_kernel.width() / 2,
                               y + r_precond_kernel.y - precond_kernel.height() / 2,
			                   c) *
	                precond_kernel_func(r_precond_kernel.x, r_precond_kernel.y);
    Func z0("z0");
    z0(x, y, c) = 0.f;
    z0(x, y, c) += Pr0(x - r_precond_kernel.x + precond_kernel.width() / 2,
                       y - r_precond_kernel.y + precond_kernel.height() / 2,
        	  	       c) *
  	               precond_kernel_func(r_precond_kernel.x,
                                       r_precond_kernel.y);

    Func p0("p0");
    p0(x, y, c) = z0(x, y, c);
    Func xrp("xrp");
    xrp(x, y, c, n) = 0.f;
    xrp(x, y, c, 0) = clamped_x0(x, y, c);
    xrp(x, y, c, 1) = r0(x, y, c);
    xrp(x, y, c, 2) = p0(x, y, c);
    xrp(x, y, c, 3) = z0(x, y, c);

    std::map<std::string, Func> func_map;
    func_map["x0_func"] = x0_func;
    func_map["reg_kernel_weights_func"] = reg_kernel_weights_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["reg_targets_func"] = reg_targets_func;
    func_map["precond_kernel_func"] = precond_kernel_func;
    func_map["w_kernel_func"] = w_kernel_func;
    func_map["w_reg_kernels_func"] = w_reg_kernels_func;
    func_map["reg_targets_func"] = reg_targets_func;
    func_map["xrp"] = xrp;
    return func_map;
}

template <typename I0,
          typename I1,
          typename I2,
          typename I3,
          typename I4,
          typename I5,
          typename I6>
std::map<std::string, Func> deconv_cg_iter(
        const I0 &xrp,
        const I1 &kernel,
        const I2 &reg_kernel_weights,
        const I3 &reg_kernels,
        const I4 &precond_kernel,
        const I5 &w_kernel,
        const I6 &w_reg_kernels) {
    // A single iteration of conjugate gradient, takes X, R, P, Z and updates them
    Func xrp_func("xrp_func");
    xrp_func(x, y, c, n) = xrp(x, y, c, n);
    Func reg_kernel_weights_func("reg_kernel_weights_func");
    reg_kernel_weights_func(n) = reg_kernel_weights(n);
    Func reg_kernels_func("reg_kernels_func");
    reg_kernels_func(x, y, n) = reg_kernels(x, y, n);
    Func precond_kernel_func("precond_kernel_func");
    precond_kernel_func(x, y) = precond_kernel(x, y);
    Func w_kernel_func("w_kernel_func");
    w_kernel_func(x, y, c) = w_kernel(x, y, c);
    Func w_reg_kernels_func("w_reg_kernels_func");
    w_reg_kernels_func(x, y, c, n) = w_reg_kernels(x, y, c, n);

    RDom r_image(0, xrp.width(), 0, xrp.height(), 0, xrp.channels());
    RDom r_kernel(kernel);
    Func xrp_clamped = BoundaryConditions::repeat_edge(xrp_func,
                {{Expr(0), Expr(xrp.width())},
                 {Expr(0), Expr(xrp.height())},
                 {Expr(), Expr()},
                 {Expr(), Expr()}});
    Func clamped_w_kernel = BoundaryConditions::repeat_edge(w_kernel_func,
                {{Expr(0), Expr(xrp.width())},
                 {Expr(0), Expr(xrp.height())},
                 {Expr(), Expr()}});
    Func clamped_w_reg_kernels = BoundaryConditions::repeat_edge(w_reg_kernels_func,
                {{Expr(0), Expr(xrp.width())},
                 {Expr(0), Expr(xrp.height())},
                 {Expr(), Expr()},
                 {Expr(), Expr()}});
    // Extract input
    Func xk("xk");
    xk(x, y, c) = xrp_clamped(x, y, c, 0);
    Func r("r");
    r(x, y, c) = xrp_clamped(x, y, c, 1);
    Func p("p");
    p(x, y, c) = xrp_clamped(x, y, c, 2);
    Func z("z");
    z(x, y, c) = xrp_clamped(x, y, c, 3);

    Func rTz("rTz");
    // alpha = r^T * z / p^T A^T W A p
    rTz() = 0.f;
    rTz() += r(r_image.x, r_image.y, r_image.z) *
             z(r_image.x, r_image.y, r_image.z);
    Func Kp("Kp");
    Kp(x, y, c) = 0.f;
    Kp(x, y, c) += p(x + r_kernel.x - kernel.width()  / 2,
                     y + r_kernel.y - kernel.height() / 2,
                     c) *
                   kernel(r_kernel.x, r_kernel.y);
    Func WKp("WKp");
    WKp(x, y, c) = Kp(x, y, c) * clamped_w_kernel(x, y, c);
    Func KTWKp("K^TWKp");
    KTWKp(x, y, c) = 0.f;
    KTWKp(x, y, c) += WKp(x - r_kernel.x + kernel.width()  / 2,
                          y - r_kernel.y + kernel.height() / 2,
                          c) *
                       kernel(r_kernel.x, r_kernel.y);
    RDom r_reg_kernel_xy(0, reg_kernels.width(), 0, reg_kernels.height());
    RDom r_reg_kernel_z(0, reg_kernels.channels());
    Func rKp("rKp");
    rKp(x, y, c, n) = 0.f;
    rKp(x, y, c, n) += p(x + r_reg_kernel_xy.x - reg_kernels.width()  / 2,
                         y + r_reg_kernel_xy.y - reg_kernels.height() / 2,
                         c) *
                       reg_kernels_func(r_reg_kernel_xy.x, r_reg_kernel_xy.y, n);
    Func WrKp("WrKp");
    WrKp(x, y, c, n) = rKp(x, y, c, n) * clamped_w_reg_kernels(x, y, c, n);
    Func rKTWrKp("rK^TWrKp");
    rKTWrKp(x, y, c, n) = 0.f;
    rKTWrKp(x, y, c, n) += WrKp(x - r_reg_kernel_xy.x + reg_kernels.width()  / 2,
                                y - r_reg_kernel_xy.y + reg_kernels.height() / 2,
                                c,
                                n) *
                           reg_kernels_func(r_reg_kernel_xy.x,
                                            r_reg_kernel_xy.y,
                                            n);
    Func ATWAp("A^TWAp");
    ATWAp(x, y, c) = KTWKp(x, y, c);
    ATWAp(x, y, c) += rKTWrKp(x, y, c, r_reg_kernel_z.x) *
                      reg_kernel_weights_func(r_reg_kernel_z.x) *
                      reg_kernel_weights_func(r_reg_kernel_z.x);
    Func pTATWAp("p^TA^TWAp");
    pTATWAp() = 0.f;
    pTATWAp() += p(r_image.x, r_image.y, r_image.z) *
                 ATWAp(r_image.x, r_image.y, r_image.z);

    Func alpha("alpha");
    alpha() = rTz() / pTATWAp();
    // x = x + alpha * p
    Func next_x("next_x");
    next_x(x, y, c) = xk(x, y, c) + alpha() * p(x, y, c);
    // r = r - alpha * A^TAp
    Func next_r("next_r");
    next_r(x, y, c) = r(x, y, c) - alpha() * ATWAp(x, y, c);

    RDom r_precond_kernel(precond_kernel);
    Func Pr("Pr");
    Pr(x, y, c) = 0.f;
    Pr(x, y, c) += r(x + r_precond_kernel.x - precond_kernel.width() / 2,
                     y + r_precond_kernel.y - precond_kernel.height() / 2,
             	     c) *
                   precond_kernel_func(r_precond_kernel.x, r_precond_kernel.y);
    Func next_z("next_z");
    next_z(x, y, c) = 0.f;
    next_z(x, y, c) += Pr(x - r_precond_kernel.x + precond_kernel.width() / 2,
                          y - r_precond_kernel.y + precond_kernel.height() / 2,
        	  	          c) *
  	                   precond_kernel_func(r_precond_kernel.x,
                                           r_precond_kernel.y);

    // beta = nextZ^TnextR / r^Tr
    Func nRTnZ("nRTnZ");
    nRTnZ() = 0.f;
    nRTnZ() += next_r(r_image.x, r_image.y, r_image.z) *
               next_z(r_image.x, r_image.y, r_image.z);
    Func beta("beta");
    beta() = nRTnZ() / rTz();
    Func next_p("next_p");
    next_p(x, y, c) = next_z(x, y, c) + beta() * p(x, y, c);

    Func next_xrp("next_xrp");
    next_xrp(x, y, c, n) = 0.f;
    next_xrp(x, y, c, 0) = next_x(x, y, c);
    next_xrp(x, y, c, 1) = next_r(x, y, c);
    next_xrp(x, y, c, 2) = next_p(x, y, c);
    next_xrp(x, y, c, 3) = next_z(x, y, c);

    std::map<std::string, Func> func_map;
    func_map["reg_kernel_weights_func"] = reg_kernel_weights_func;
    func_map["reg_kernels_func"] = reg_kernels_func;
    func_map["precond_kernel_func"] = precond_kernel_func;
    func_map["w_kernel_func"] = w_kernel_func;
    func_map["w_reg_kernels_func"] = w_reg_kernels_func;
    func_map["xrp_func"] = xrp_func;
    func_map["next_xrp"] = next_xrp;
    return func_map;
}


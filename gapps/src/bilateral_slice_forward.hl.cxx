#include "Halide.h"

using Halide::Func;
using Halide::Expr;
using Halide::Var;
using Halide::undef;
using Halide::BoundaryConditions::repeat_edge;

namespace gradient_apps {

class BilateraSliceForwardGenerator
  : public Halide::Generator<BilateraSliceForwardGenerator> {
private:
  Var x, y, z, c, n;

public:
  Input<Buffer<float>> grid{"input", 5};
  Input<Buffer<float>> guide{"guide", 3};
  Output<Buffer<float>> output{"output", 4};


  void generate() {
    Func cgrid("clamped_grid");
    cgrid = Halide::BoundaryConditions::repeat_edge(grid);

    Func cguide("clamped_guide");
    cguide = Halide::BoundaryConditions::repeat_edge(guide);

    Expr gw = grid.dim(0).extent();
    Expr gh = grid.dim(1).extent();
    Expr gd = grid.dim(2).extent();
    Expr nc = grid.dim(3).extent();
    Expr w = guide.dim(0).extent();
    Expr h = guide.dim(1).extent();

    // Coordinates in the bilateral grid
    Expr gx = (x+0.5f)*gw/(1.0f*w);
    Expr gy = (y+0.5f)*gh/(1.0f*h);
    Expr gz = clamp(cguide(x, y, n), 0.0f, 1.0f)*gd;

    // Floor voxel
    Expr fx = cast<int>(floor(gx-0.5f));
    Expr fy = cast<int>(floor(gy-0.5f));
    Expr fz = cast<int>(floor(gz-0.5f));

    // Ceil voxel
    Expr cx = fx+1;
    Expr cy = fy+1;
    Expr cz = fz+1;

    // weights
    Expr wx = gx-0.5f - fx;
    Expr wy = gy-0.5f - fy;
    Expr wz = gz-0.5f - fz;

    // Tri-linear interpolation
    Func interp_y("interp_y");
    Func interp_x("interp_x");
    interp_y(x, y, z, c, n) = cgrid(x, fy, z, c, n)*(1-wy) + cgrid(x, cy, z, c, n)*wy;
    interp_x(x, y, z, c, n) = interp_y(fx, y, z, c, n)*(1-wx) + interp_y(cx, y, z, c, n)*wx;
    output(x, y, c, n) = interp_x(x, y, fz, c, n)*(1-wz) + interp_x(x, y, cz, c, n)*wz;
  }
  void schedule() {
    int parallel_sz = 2;
    int vector_w = 8;
    output
        .compute_root()
        .parallel(n)
        // .parallel(c, parallel_sz)
        .vectorize(x, vector_w);

    std::vector<Func> outputs = {
      output
    };
  }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateraSliceForwardGenerator, bilateral_slice_forward)

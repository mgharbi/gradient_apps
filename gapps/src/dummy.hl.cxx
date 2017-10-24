#include "Halide.h"

using Halide::Func;
using Halide::Expr;
using Halide::Var;
using Halide::undef;
using Halide::BoundaryConditions::repeat_edge;

namespace gradient_apps {

class DummyGenerator : public Halide::Generator<DummyGenerator> {
private:
  Var x, y, c, n;

public:
  Input<Buffer<float>> input{"input", 4};
  Output<Buffer<float>> output{"output", 4};

  void generate() {
    Func clamped("clamped");
    clamped = Halide::BoundaryConditions::repeat_edge(input);

    output(x, y, c, n) = input(x, y, c, n)*cast<float>(c);

  }
  void schedule() {
    int parallel_sz = 2;
    int vector_w = 8;
    output
        .compute_root()
        .parallel(c, parallel_sz)
        .vectorize(x, vector_w);
  }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(gradient_apps::DummyGenerator, dummy)

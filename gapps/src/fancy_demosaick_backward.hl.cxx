#include "algorithms/fancy_demosaick.h"
#include <vector>

#include "gradient_helpers.h"

namespace gradient_apps {

class FancyDemosaickBackwardGenerator : 
  public Generator<FancyDemosaickBackwardGenerator> {
public:
  Input<Buffer<float>>  cfa{"cfa", 3};
  Input<Func[2]> weights{"weights", Float(32), 1};
  Input<Func[2]> weights2d{"weights2d", Float(32), 2};
  Input<Buffer<float>> d_output{"output", 4};

  Output<Func[2]> d_weights{"d_weights", Float(32), 1};
  Output<Func[2]> d_weights2d{"d_weights2d", Float(32), 2};

  void generate() {
    std::map<std::string, Func> f = fancy_demosaick(
        cfa, weights, weights2d);
    Func output = f["output"];
    // output(x, y, c, n) = f_output(x, y, c, n);

    Derivative d = propagate_adjoints(output, d_output,
        {
          {d_output.dim(0).min(), d_output.dim(0).max()},
          {d_output.dim(1).min(), d_output.dim(1).max()},
          {d_output.dim(2).min(), d_output.dim(2).max()},
          {d_output.dim(3).min(), d_output.dim(3).max()},
        });

    std::vector<Func> funcs;
    for(int i = 0; i < 2; ++i) {
      assign_gradient(d, weights[i], d_weights[i]);
      funcs.push_back(d_weights[i]);
    }
    for(int i = 0; i < 2; ++i) {
      assign_gradient(d, weights2d[i], d_weights2d[i]);
      funcs.push_back(d_weights2d[i]);
    }

    SimpleAutoscheduleOptions options;
    options.gpu = get_target().has_gpu_feature();
    std::set<std::string> dont_inline = {};
    int bs = 1;
  //   simple_autoschedule(funcs,
  //       {
  //       {"input.min.0", 0},
  //       {"input.min.1", 0},
  //       {"input.min.2", 0},
  //       {"input.min.2", 0},
  //       {"input.extent.0", 224},
  //       {"input.extent.1", 224},
  //       {"input.extent.2", 3},
  //       {"input.extent.3", bs},
  //       {"weights_0.min.0", 0},
  //       {"weights_0.extent.0", 3},
  //       {"weights_1.min.0", 0},
  //       {"weights_1.extent.0", 3},
  //       {"weights2d_0.min.0", 0},
  //       {"weights2d_0.min.1", 0},
  //       {"weights2d_0.extent.0", 4},
  //       {"weights2d_0.extent.1", 4},
  //       {"weights2d_1.min.0", 0},
  //       {"weights2d_1.min.1", 0},
  //       {"weights2d_1.extent.0", 4},
  //       {"weights2d_1.extent.1", 4},
  //       },
  //       {
  //       {{0, 223}, {0, 223}, {0, 2}, {0, bs-1}},
  //       },
  //       options,
  //       dont_inline);
  }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::FancyDemosaickBackwardGenerator, 
    fancy_demosaick_backward)

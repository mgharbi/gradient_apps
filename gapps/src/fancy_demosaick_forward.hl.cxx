#include "algorithms/fancy_demosaick.h"
#include <vector>

#include "gradient_helpers.h"

namespace gradient_apps {

class FancyDemosaickForwardGenerator : 
  public Generator<FancyDemosaickForwardGenerator> {
public:
  Input<Buffer<float>>  cfa{"cfa", 3};
  Input<Func[2]> weights{"weights", Float(32), 1};
  Input<Func[2]> weights2d{"weights2d", Float(32), 2};
  Output<Buffer<float>> output{"output", 4};

  void generate() {
    std::map<std::string, Func> f = fancy_demosaick(
        cfa, weights, weights2d);
    Func f_output = f["output"];
    output(x, y, c, n) = f_output(x, y, c, n);

    SimpleAutoscheduleOptions options;
    options.gpu = get_target().has_gpu_feature();
    std::set<std::string> dont_inline = {};
    std::vector<Func> funcs{output};
    int bs = 1;
    std::cout << "forward autoschedule" << std::endl;
    simple_autoschedule(funcs,
        {
        {"input.min.0", 0},
        {"input.min.1", 0},
        {"input.min.2", 0},
        {"input.min.2", 0},
        {"input.extent.0", 224},
        {"input.extent.1", 224},
        {"input.extent.2", 3},
        {"input.extent.3", bs},
        {"weights_0.min.0", 0},
        {"weights_0.extent.0", 3},
        {"weights_1.min.0", 0},
        {"weights_1.extent.0", 3},
        {"weights2d_0.min.0", 0},
        {"weights2d_0.min.1", 0},
        {"weights2d_0.extent.0", 4},
        {"weights2d_0.extent.1", 4},
        {"weights2d_1.min.0", 0},
        {"weights2d_1.min.1", 0},
        {"weights2d_1.extent.0", 4},
        {"weights2d_1.extent.1", 4},
        },
        {
        {{0, 223}, {0, 223}, {0, 2}, {0, bs-1}},
        },
        options,
        dont_inline);
  }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::FancyDemosaickForwardGenerator, 
    fancy_demosaick_forward)

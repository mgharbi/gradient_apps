#include "algorithms/fancy_demosaick.h"
#include <vector>

#include "gradient_helpers.h"

namespace gradient_apps {

class FancyDemosaickForwardGenerator : 
  public Generator<FancyDemosaickForwardGenerator> {
public:
  Input<Buffer<float>>  cfa{"cfa", 3};
  Input<Func[n_w]> weights{"weights", Float(32), 1};
  Input<Func[n_w2]> weights2d{"weights2d", Float(32), 2};
  Input<Func[n_w3]> weights3d{"weights3d", Float(32), 3};
  Input<Func[n_w4]> weights4d{"weights4d", Float(32), 4};
  Output<Buffer<float>> output{"output", 4};

  void generate() {
    std::map<std::string, Func> f = fancy_demosaick(
        cfa, weights, weights2d, weights3d, weights4d);
    Func f_output = f["output"];
    output(x, y, c, n) = f_output(x, y, c, n);

    SimpleAutoscheduleOptions options;
    options.gpu = get_target().has_gpu_feature();
    std::set<std::string> dont_inline = {};
    std::vector<Func> funcs{output};
    std::cout << "forward autoschedule" << std::endl;
    simple_autoschedule(funcs,
        {
        {"cfa.min.0", 0},
        {"cfa.min.1", 0},
        {"cfa.min.2", 0},
        {"cfa.min.2", 0},
        {"cfa.extent.0", 128},
        {"cfa.extent.1", 128},
        {"cfa.extent.2", 3},
        {"cfa.extent.3", 1},
        {"weights3d_0.min.0", 0},
        {"weights3d_0.min.1", 0},
        {"weights3d_0.min.2", 0},
        {"weights3d_0.extent.0", 7},
        {"weights3d_0.extent.1", 7},
        {"weights3d_0.extent.2", 9},
        {"weights4d_0.min.0", 0},
        {"weights4d_0.min.1", 0},
        {"weights4d_0.min.2", 0},
        {"weights4d_0.min.3", 0},
        {"weights4d_0.extent.0", 7},
        {"weights4d_0.extent.1", 7},
        {"weights4d_0.extent.2", 4},
        {"weights4d_0.extent.3", 9},
        },
        {
        {{0, 127}, {0, 127}, {0, 2}, {0, 0}},
        },
        options,
        dont_inline);
  }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::FancyDemosaickForwardGenerator, 
    fancy_demosaick_forward)

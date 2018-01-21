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
  Output<Buffer<float>> output{"output", 4};

  void generate() {
    std::map<std::string, Func> f = fancy_demosaick(
        cfa, weights, weights2d, weights3d);
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
        {"weights_0.min.0", 0},
        {"weights_0.extent.0", 5},
        {"weights_1.min.0", 0},
        {"weights_1.extent.0", 5},
        {"weights_2.min.0", 0},
        {"weights_2.extent.0", 5},
        {"weights_3.min.0", 0},
        {"weights_3.extent.0", 5},
        {"weights_4.min.0", 0},
        {"weights_4.extent.0", 5},
        {"weights_5.min.0", 0},
        {"weights_5.extent.0", 5},
        {"weights2d_0.min.0", 0},
        {"weights2d_0.min.1", 0},
        {"weights2d_0.extent.0", 4},
        {"weights2d_0.extent.1", 4},
        {"weights2d_1.min.0", 0},
        {"weights2d_1.min.1", 0},
        {"weights2d_1.extent.0", 4},
        {"weights2d_1.extent.1", 4},
        {"weights2d_2.min.0", 0},
        {"weights2d_2.min.1", 0},
        {"weights2d_2.extent.0", 4},
        {"weights2d_2.extent.1", 4},
        {"weights2d_3.min.0", 0},
        {"weights2d_3.min.1", 0},
        {"weights2d_3.extent.0", 4},
        {"weights2d_3.extent.1", 4},
        {"weights2d_4.min.0", 0},
        {"weights2d_4.min.1", 0},
        {"weights2d_4.extent.0", 5},
        {"weights2d_4.extent.1", 2},
        {"weights2d_5.min.0", 0},
        {"weights2d_5.min.1", 0},
        {"weights2d_5.extent.0", 5},
        {"weights2d_5.extent.1", 2},
        {"weights2d_6.min.0", 0},
        {"weights2d_6.min.1", 0},
        {"weights2d_6.extent.0", 5},
        {"weights2d_6.extent.1", 2},
        {"weights3d_0.min.0", 0},
        {"weights3d_0.min.1", 0},
        {"weights3d_0.min.2", 0},
        {"weights3d_0.extent.0", 4},
        {"weights3d_0.extent.1", 4},
        {"weights3d_0.extent.2", 2},
        {"weights3d_1.min.0", 0},
        {"weights3d_1.min.1", 0},
        {"weights3d_1.min.2", 0},
        {"weights3d_1.extent.0", 4},
        {"weights3d_1.extent.1", 4},
        {"weights3d_1.extent.2", 2},
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

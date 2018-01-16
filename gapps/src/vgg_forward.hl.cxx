#include "algorithms/vgg.h"
#include <vector>

#include "gradient_helpers.h"

namespace gradient_apps {

class VGGForwardGenerator : 
  public Generator<VGGForwardGenerator> {
public:
  Input<Buffer<float>>  input{"input", 4};
  Input<Func[13]>  weights{"weights", Float(32), 4};
  Input<Func[3]>  fc_weights{"fc_weights", Float(32), 2};
  Input<Func[16]>  biases{"biases", Float(32), 1};
  Output<Buffer<float>> output{"output", 2};

  void generate() {
    std::map<std::string, Func> func_map = vgg(
        input, weights, fc_weights, biases);
    Func f_output = func_map["output"];
    output(c, n) = f_output(c, n);

    SimpleAutoscheduleOptions options;
    options.gpu = get_target().has_gpu_feature();

    std::set<std::string> dont_inline = {};

    std::vector<Func> funcs{output};

    int bs = 1;

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
        {"weights_0.min.1", 0},
        {"weights_0.min.2", 0},
        {"weights_0.min.3", 0},
        {"weights_0.extent.0", 3},
        {"weights_0.extent.1", 3},
        {"weights_0.extent.2", 3},
        {"weights_0.extent.3", 64},
        {"weights_1.min.0", 0},
        {"weights_1.min.1", 0},
        {"weights_1.min.2", 0},
        {"weights_1.min.3", 0},
        {"weights_1.extent.0", 3},
        {"weights_1.extent.1", 3},
        {"weights_1.extent.2", 64},
        {"weights_1.extent.3", 64},
        {"weights_2.min.0", 0},
        {"weights_2.min.1", 0},
        {"weights_2.min.2", 0},
        {"weights_2.min.3", 0},
        {"weights_2.extent.0", 3},
        {"weights_2.extent.1", 3},
        {"weights_2.extent.2", 64},
        {"weights_2.extent.3", 128},
        {"weights_3.min.0", 0},
        {"weights_3.min.1", 0},
        {"weights_3.min.2", 0},
        {"weights_3.min.3", 0},
        {"weights_3.extent.0", 3},
        {"weights_3.extent.1", 3},
        {"weights_3.extent.2", 128},
        {"weights_3.extent.3", 128},
        {"weights_4.min.0", 0},
        {"weights_4.min.1", 0},
        {"weights_4.min.2", 0},
        {"weights_4.min.3", 0},
        {"weights_4.extent.0", 3},
        {"weights_4.extent.1", 3},
        {"weights_4.extent.2", 128},
        {"weights_4.extent.3", 256},
        {"weights_5.min.0", 0},
        {"weights_5.min.1", 0},
        {"weights_5.min.2", 0},
        {"weights_5.min.3", 0},
        {"weights_5.extent.0", 3},
        {"weights_5.extent.1", 3},
        {"weights_5.extent.2", 256},
        {"weights_5.extent.3", 256},
        {"weights_6.min.0", 0},
        {"weights_6.min.1", 0},
        {"weights_6.min.2", 0},
        {"weights_6.min.3", 0},
        {"weights_6.extent.0", 3},
        {"weights_6.extent.1", 3},
        {"weights_6.extent.2", 256},
        {"weights_6.extent.3", 256},
        {"weights_7.min.0", 0},
        {"weights_7.min.1", 0},
        {"weights_7.min.2", 0},
        {"weights_7.min.3", 0},
        {"weights_7.extent.0", 3},
        {"weights_7.extent.1", 3},
        {"weights_7.extent.2", 256},
        {"weights_7.extent.3", 512},
        {"weights_8.min.0", 0},
        {"weights_8.min.1", 0},
        {"weights_8.min.2", 0},
        {"weights_8.min.3", 0},
        {"weights_8.extent.0", 3},
        {"weights_8.extent.1", 3},
        {"weights_8.extent.2", 512},
        {"weights_8.extent.3", 512},
        {"weights_9.min.0", 0},
        {"weights_9.min.1", 0},
        {"weights_9.min.2", 0},
        {"weights_9.min.3", 0},
        {"weights_9.extent.0", 3},
        {"weights_9.extent.1", 3},
        {"weights_9.extent.2", 512},
        {"weights_9.extent.3", 512},

        {"weights_10.min.0", 0},
        {"weights_10.min.1", 0},
        {"weights_10.min.2", 0},
        {"weights_10.min.3", 0},
        {"weights_10.extent.0", 3},
        {"weights_10.extent.1", 3},
        {"weights_10.extent.2", 512},
        {"weights_10.extent.3", 512},
        {"weights_11.min.0", 0},
        {"weights_11.min.1", 0},
        {"weights_11.min.2", 0},
        {"weights_11.min.3", 0},
        {"weights_11.extent.0", 3},
        {"weights_11.extent.1", 3},
        {"weights_11.extent.2", 512},
        {"weights_11.extent.3", 512},
        {"weights_12.min.0", 0},
        {"weights_12.min.1", 0},
        {"weights_12.min.2", 0},
        {"weights_12.min.3", 0},
        {"weights_12.extent.0", 3},
        {"weights_12.extent.1", 3},
        {"weights_12.extent.2", 512},
        {"weights_12.extent.3", 512},
        {"fc_weights_0.min.0", 0},
        {"fc_weights_0.min.1", 0},
        {"fc_weights_0.extent.0", 512},
        {"fc_weights_0.extent.1", 4096},
        {"fc_weights_1.min.0", 0},
        {"fc_weights_1.min.1", 0},
        {"fc_weights_1.extent.0", 512},
        {"fc_weights_1.extent.1", 4096},
        {"fc_weights_2.min.0", 0},
        {"fc_weights_2.min.1", 0},
        {"fc_weights_2.extent.0", 512},
        {"fc_weights_2.extent.1", 4096},
        },
        {
          {{0, 999}, {0, bs-1}},
        },
        options,
        dont_inline);
  }

};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::VGGForwardGenerator, 
    vgg_forward)

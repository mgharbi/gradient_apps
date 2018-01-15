#include "algorithms/vgg.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class VGGForwardGenerator : 
  public Generator<VGGForwardGenerator> {
public:
  Input<Buffer<float>>  inputs{"input", 4};
  Input<Buffer<float>>  inputs{"input", 4};
  Output<Buffer<float>> loss{"output", 4};

  void generate() {
    std::map<std::string, Func> func_map = vgg(
        inputs, homographies, reconstructed, gradient_weight);
    Func f_loss = func_map["loss"];
    Func f_reproj = func_map["reproj"];
    loss(x) = f_loss(x);
    reproj_error(x, y, n) = f_reproj(x, y, n);

    SimpleAutoscheduleOptions options;
    options.gpu = get_target().has_gpu_feature();
    Func loss_func = loss;
    Func reproj_func = reproj_error;

    std::set<std::string> dont_inline = {};

    std::vector<Func> funcs{loss_func, reproj_func};

    simple_autoschedule(funcs,
        {
        {"inputs.min.0", 0},
        {"inputs.min.1", 0},
        {"inputs.min.2", 0},
        {"inputs.extent.0", 256},
        {"inputs.extent.1", 256},
        {"inputs.extent.2", 5},
        // {"confidence.min.0", 0},
        // {"confidence.min.1", 0},
        // {"confidence.min.2", 0},
        // {"confidence.extent.0", 256},
        // {"confidence.extent.1", 256},
        // {"confidence.extent.2", 5},
        {"homographies.min.0", 0},
        {"homographies.min.1", 0},
        {"homographies.extent.0", 8},
        {"homographies.extent.1", 5},
        {"reconstructed.min.0", 0},
        {"reconstructed.min.1", 0},
        {"reconstructed.min.2", 0},
        {"reconstructed.extent.0", 256},
        {"reconstructed.extent.1", 256},
        {"reconstructed.extent.2", 3},
        {"gradient_weight.min.0", 0},
        {"gradient_weight.extent.0", 0},
        },
        {
          {{0, 1}},
          {{0, 255}, {0, 255}, {0, 5}},
        },
        options,
        dont_inline);
  }

};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::VGGForwardGenerator, 
    vgg_forward)

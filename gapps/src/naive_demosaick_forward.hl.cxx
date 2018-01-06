#include "algorithms/naive_demosaick.h"

namespace gradient_apps {

class NaiveDemosaickForwardGenerator : public Generator<NaiveDemosaickForwardGenerator> {
public:
  Input<Buffer<float>>  mosaick{"mosaick", 3};       // x, y, n
  Output<Buffer<float>> output{"output", 4};     // x, y, 3, n


  void generate() {
    std::map<std::string, Func> func_map = naive_demosaick(mosaick);
    Func f_output = func_map["output"];
    output(x, y, c, n) = f_output(x, y, c, n);

    if(auto_schedule) {
    } else {
      SimpleAutoscheduleOptions options;
      options.gpu = get_target().has_gpu_feature();
      Func output_func = output;

      std::set<std::string> dont_inline = {};

      simple_autoschedule(output_func,
          {
            {"mosaick.min.0", 0},
            {"mosaick.min.1", 0},
            {"mosaick.min.2", 0},
            {"mosaick.extent.0", 64},
            {"mosaick.extent.1", 64},
            {"mosaick.extent.2", 16},
          },
          {{0, 63},
            {0, 63},
            {0, 2},
            {0, 15}},
          options,
          dont_inline);
    }
  }

};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::NaiveDemosaickForwardGenerator, naive_demosaick_forward)

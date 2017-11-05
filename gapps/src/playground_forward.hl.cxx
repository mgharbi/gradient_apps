#include "algorithms/playground.h"

namespace gradient_apps {

class PlaygroundForwardGenerator : public Generator<PlaygroundForwardGenerator> {
public:
    Input<Buffer<float>>  input1{"input1", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  input2{"input2", 4};       // x, y, channel, batch size
    Output<Buffer<float>> output{"output", 4};     // x, y, channel, batch size


    void generate() {
        std::map<std::string, Func> func_map = playground(
            input1, input2);
        Func f_output = func_map["output"];
        output(x, y, co, n) = f_output(x, y, co, n);

        if(auto_schedule) {
          printf("Autoscheduling AHD demosaicking forward\n");
          int est_h = 512;
          int est_w = 512;
          input1.dim(0).set_bounds_estimate(0, est_w);
          input1.dim(1).set_bounds_estimate(0, est_h);
          input1.dim(2).set_bounds_estimate(0, 3);
          input1.dim(3).set_bounds_estimate(0, 16);

          input2.dim(0).set_bounds_estimate(0, est_w);
          input2.dim(1).set_bounds_estimate(0, est_h);
          input2.dim(2).set_bounds_estimate(0, 3);
          input2.dim(3).set_bounds_estimate(0, 16);

          output.dim(0).set_bounds_estimate(0, est_w);
          output.dim(1).set_bounds_estimate(0, est_h);
          output.dim(2).set_bounds_estimate(0, 3);
          output.dim(3).set_bounds_estimate(0, 16);
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::PlaygroundForwardGenerator, playground_forward)

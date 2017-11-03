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
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::PlaygroundForwardGenerator, playground_forward)

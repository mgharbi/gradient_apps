#include "algorithms/histogram.h"

namespace gradient_apps {

class HistogramGenerator : public Generator<HistogramGenerator> {
public:
    Input<Buffer<float>>  input{"input", 2};       // x, y

    Input<int> nbins{"nbins"};
    Output<Buffer<float>> output{"output", 1};

    void generate() {
        std::map<std::string, Func> func_map = histogram(input, nbins);
        Func f_output = func_map["output"];
        output(x) = f_output(x);

        if(auto_schedule) {
          printf("Autoscheduling forward\n");
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::HistogramGenerator, histogram_forward)

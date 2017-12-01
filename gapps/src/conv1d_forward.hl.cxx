#include "algorithms/conv1d.h"

namespace gradient_apps {

class Conv1dForwardGenerator : public Generator<Conv1dForwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 3};
    Input<Buffer<float>>  filter{"filter", 3};
    Output<Buffer<float>> output{"output", 3};

    void generate() {
        std::map<std::string, Func> func_map = conv1d(
            input, filter);

        Func f_output = func_map["output"];
        output(x, co, n) = f_output(x, co, n);

        if(auto_schedule) {
        } else {
          if (get_target().has_gpu_feature()) {
          } else {

            print_func(output);


            Var nc("nc");
            output
              .compute_root()
              .fuse(n, co, nc)
              .parallel(nc)
              .vectorize(x, 8)
              ;
            f_output
              .compute_at(output, x)
              .vectorize(x, 8);
            f_output
              .update()
              .vectorize(x, 8);
          }
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv1dForwardGenerator, conv1d_forward)

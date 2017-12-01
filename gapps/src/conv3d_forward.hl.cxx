#include "algorithms/conv3d.h"

namespace gradient_apps {

class Conv3dForwardGenerator : public Generator<Conv3dForwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 5};
    Input<Buffer<float>>  filter{"filter", 5};
    Output<Buffer<float>> output{"output", 5};

    void generate() {
        std::map<std::string, Func> func_map = conv3d(
            input, filter);
        // printf("\ninput deps:\n\n");
        // print_func(func_map["input"]);

        Func f_output = func_map["output"];
        output(x, y, z, co, n) = f_output(x, y, z, co, n);


        if(auto_schedule) {
        } else {
          if (get_target().has_gpu_feature()) {
          } else {
            output
              .compute_root()
              .parallel(n)
              .parallel(co)
              .parallel(z)
              .vectorize(x, 8)
              ;
          }
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv3dForwardGenerator, conv3d_forward)

#include "algorithms/bilateral_layer.h"

namespace gradient_apps {

class BilateralLayerForwardGenerator : public Generator<BilateralLayerForwardGenerator> {
public:
    Input<int> sigma_x{"sigma_x"}; // block_size in x
    Input<int> sigma_y{"sigma_y"}; // block_size in y
    Input<int> sigma_z{"sigma_z"}; // number of guide discrete levels

    Input<Buffer<float>>  input{"input", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};       // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};     // x, y, z, input channel, output channel

    Output<Buffer<float>> output{"output", 4};     // x, y, channel, batch size

    void generate() {
        std::map<std::string, Func> func_map = bilateral_layer(
            input, guide, filter, sigma_x, sigma_y, sigma_z);
        Func f_output = func_map["output"];
        output(x, y, co, n) = f_output(x, y, co, n);

        if(auto_schedule) {
          printf("Autoscheduling bilateral_layer forward\n");

          int est_bsize = 1;
          int est_h = 128;
          int est_w = 128;
          int est_ci = 3;
          int est_co = 3;
          int est_kh = 3;
          int est_kw = 3;
          int est_kd = 3;
          input.dim(0).set_bounds_estimate(0, est_w);
          input.dim(1).set_bounds_estimate(0, est_h);
          input.dim(2).set_bounds_estimate(0, est_ci);
          input.dim(3).set_bounds_estimate(0, est_bsize);

          guide.dim(0).set_bounds_estimate(0, est_w);
          guide.dim(1).set_bounds_estimate(0, est_h);
          guide.dim(2).set_bounds_estimate(0, est_bsize);

          filter.dim(0).set_bounds_estimate(0, est_kw);
          filter.dim(1).set_bounds_estimate(0, est_kh);
          filter.dim(2).set_bounds_estimate(0, est_kd);
          filter.dim(3).set_bounds_estimate(0, est_ci);
          filter.dim(4).set_bounds_estimate(0, est_co);

          output
            .estimate(x, 0, est_w)
            .estimate(y, 0, est_h)
            .estimate(co, 0, est_co)
            .estimate(n, 0, est_bsize)
            ;
        } else {
          if (get_target().has_gpu_feature()) {
            Var xi, yi;
            func_map["grid"]
              .compute_root()
              .gpu_tile(x, y, xi, yi, 8, 8);
              ;
            func_map["grid"]
              .update(0)
              .gpu_tile(x, y, xi, yi, 8, 8);
              ;
            func_map["grid"]
              .update(1)
              .gpu_tile(x, y, xi, yi, 8, 8);
              ;
            func_map["conv"]
              .compute_root()
              .gpu_tile(x, y, xi, yi, 8, 8);
              ;
            func_map["conv"]
              .update(0)
              .gpu_tile(x, y, xi, yi, 8, 8);
              ;
          } else {
            func_map["grid"]
              .compute_root()
              .parallel(n)
              .parallel(ci)
              .parallel(z)
              .vectorize(x, 8)
              ;
            func_map["grid"]
              .update(0)
              .parallel(n)
              .parallel(ci)
              .parallel(y)
              .vectorize(x, 8)
              ;
            func_map["grid"]
              .update(1)
              .parallel(n)
              .parallel(ci)
              .parallel(y)
              .vectorize(x, 8)
              ;
            func_map["conv"]
              .compute_root()
              .parallel(n)
              .parallel(co)
              .parallel(z)
              .vectorize(x, 8)
              ;
            func_map["conv"]
              .update(0)
              .parallel(n)
              .parallel(co)
              .parallel(z)
              .vectorize(x, 8)
              ;
            output
              .compute_root()
              .parallel(n)
              .parallel(co)
              .parallel(y)
              .vectorize(x, 8)
              ;
          }
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralLayerForwardGenerator, bilateral_layer_forward)

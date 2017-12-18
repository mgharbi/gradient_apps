#include "algorithms/learnable_demosaick.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class LearnableDemosaickForwardGenerator : public Generator<LearnableDemosaickForwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 3};
    Input<Buffer<float>>  sel_filts{"selection_filters", 3};
    Input<Buffer<float>>  green_filts{"green_filters", 3};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        std::map<std::string, Func> func_map = learnable_demosaick(mosaick, sel_filts, green_filts);
        Func f_output = func_map["output"];
        Func normalizer = func_map["normalizer"];
        Func interp_g = func_map["interp_g"];
        Func interpolated_green = func_map["interp_green"];
        Func weights = func_map["weights"];
        output(x, y, c, n) = f_output(x, y, c, n);

        if(auto_schedule) {
        } else {
          Var xi("xi"), yi("yi"), xy("xy"), xyn("xyn");
          Var xo("xo"), yo("yo"), xyk("xyk"), xykn("xykn");
          Var xiyi("xiyi");

          if (get_target().has_gpu_feature()) {
            cerr << "gpu schedule\n";
            int ts = 64;

            func_map["sel_max"]
              .compute_root()
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              .update()
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;

            func_map["selection"]
              .compute_root()
              .reorder(k, x, y, n)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              .update()
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;

            func_map["normalizer"]
              .compute_root()
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              .update()
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;

            func_map["interp_g"]
              .compute_root()
              .reorder(k, x, y, n)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              .update()
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;

            func_map["interpolated_green"]
              .compute_root()
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              .update()
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;


            output
              .compute_root()
              .reorder(c, x, y, n)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;

          } else {
            cerr << "cpu schedule\n";
            compute_all_root(output);
            // normalizer
            //   .compute_at(output, xyn)
            //   ;
            // interp_g
            //   .compute_at(output, xyn)
            //   ;
            // interpolated_green
            //   .compute_at(output, xyn)
            //   ;
            // output
            //   .tile(x, y, xi, yi, 16, 16)
            //   .fuse(x, y, xy)
            //   .fuse(xy, n, xyn)
            //   .parallel(xyn, 8)
            //   .vectorize(xi, 8)
            //   ;
          }
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::LearnableDemosaickForwardGenerator, learnable_demosaick_forward)

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
    std::map<std::string, Func> f = vgg(
        input, weights, fc_weights, biases);
    Func f_output = f["output"];
    output(co, n) = f_output(co, n);

    bool autoschedule = false;
    if(autoschedule) {
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
    } else {
      if ( get_target().has_gpu_feature() ) {
        int ts = 512;
        int ts2 = 32;
        Var bx("bx");
        Var tx("tx");
        Var by("by");
        Var ty("ty");
        Var bc("bc");
        Var tc("tc");

        std::vector<std::string> conv_keys = {
          "conv1_1", "conv1_2", 
          "conv2_1", "conv2_2", 
          "conv3_1", "conv3_2", "conv3_3",
          "conv4_1", "conv4_2", "conv4_3",
          "conv5_1", "conv5_2", "conv5_3",
        };
        std::vector<std::string> pool_keys = {
          "pool1", "pool2", "pool3", "pool4", "pool5"
        };

        for (std::string k : conv_keys) {
          f[k].compute_root()
            .gpu_tile(x, y, co, bx, by, bc, tx, ty, tc, 4, 4, 4)
            ;
          f[k].update()
            .gpu_tile(x, y, co, bx, by, bc, tx, ty, tc, 4, 4, 4)
            ;
        }

        for (std::string k : pool_keys) {
          f[k].compute_root()
            .gpu_tile(x, y, co, bx, by, bc, tx, ty, tc, 4, 4, 4);
        }

        f["fc6"].compute_root().gpu_tile(co, bx, tx, ts);
        f["fc6"].update().gpu_tile(co, bx, tx, ts);
        f["fc7"].compute_root().gpu_tile(co, bx, tx, ts);
        f["fc7"].update().gpu_tile(co, bx, tx, ts);
        output.compute_root().gpu_tile(co, bx, tx, 100);
        f["output"].compute_at(output, bx).update().gpu_threads(co);
      } else {
        int v = 8;
        int p = 8;
        f["conv1_1"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["conv1_2"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["pool1"].fuse(n, co, co).fuse(co, y, y).compute_root()  .parallel(y).vectorize(x, v);
        f["conv2_1"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["conv2_2"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["pool2"].fuse(n, co, co).fuse(co, y, y).compute_root()  .parallel(y).vectorize(x, v);
        f["conv3_1"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["conv3_2"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["conv3_3"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["pool3"].fuse(n, co, co).fuse(co, y, y).compute_root()  .parallel(y).vectorize(x, v);
        f["conv4_1"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["conv4_2"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["conv4_3"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["pool4"].fuse(n, co, co).fuse(co, y, y).compute_root()  .parallel(y).vectorize(x, v);
        f["conv5_1"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["conv5_2"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["conv5_3"].fuse(n, co, co).fuse(co, y, y).compute_root().parallel(y).vectorize(x, v);
        f["pool5"].fuse(n, co, co).fuse(co, y, y).compute_root()  .parallel(y).vectorize(x, v);
        f["fc6"].compute_root().parallel(co);
        f["fc7"].compute_root().parallel(co);
        f["output"].compute_root().parallel(co);

        f["conv1_1"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv1_2"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv2_1"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv2_2"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv3_1"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv3_2"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv3_3"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv4_1"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv4_2"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv4_3"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv5_1"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv5_2"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["conv5_3"].update().fuse(n, co, co).fuse(co, y, y).parallel(y).vectorize(x, v);
        f["fc6"].update().parallel(co);
        f["fc7"].update().parallel(co);
        f["output"].update().parallel(co);
      }
    }
  }

};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::VGGForwardGenerator, 
    vgg_forward)

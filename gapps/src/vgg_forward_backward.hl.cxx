#include "algorithms/vgg.h"
#include <vector>

#include "gradient_helpers.h"

namespace gradient_apps {

class VGGForwardBackwardGenerator : 
  public Generator<VGGForwardBackwardGenerator> {
public:
  Input<Buffer<float>>  input{"input", 4};
  Input<Func[13]>  weights{"weights", Float(32), 4};
  Input<Func[3]>  fc_weights{"fc_weights", Float(32), 2};
  Input<Func[16]>  biases{"biases", Float(32), 1};

  Output<Buffer<float>> output{"output", 2};
  Output<Func[13]>  d_weights{"d_weights", Float(32), 4};
  Output<Func[3]>  d_fc_weights{"d_fc_weights", Float(32), 2};
  Output<Func[16]>  d_biases{"d_biases", Float(32), 1};

  void generate() {
    std::map<std::string, Func> f = vgg(
        input, weights, fc_weights, biases);
    Func f_output = f["output"];
    output(co, n) = f_output(co, n);

    // TODO: this is a stupid loss, but who cares?
    Func loss("loss");
    RDom rloss(0, 1000, 0, input.dim(3).extent());
    loss() = 0.0f;
    loss() += f_output(rloss.x, rloss.y);

    // TODO: backprop
    Derivative d = propagate_adjoints(loss);

    for(int i = 0; i < 13; ++i) {
      d_weights[i](x, y, ci, co) = 0.0f;
      // d(weights[i]);
      // d_weights[i] = d(weights[i]);
    }

    for(int i = 0; i < 3; ++i) {
      d_fc_weights[i] = d(fc_weights[i]);
    }

    for(int i = 0; i < 16; ++i) {
      d_biases[i](x) = 0.0f;
    }
    

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
      if ( get_target().has_gpu_feature() ) {
        int ts = 512;
        int ts2 = 32;
        Var bx("bx");
        Var tx("tx");
        Var by("by");
        Var ty("ty");
        Var bc("bc");
        Var tc("tc");

        for (std::string k : conv_keys) {
          f[k].compute_root()
            .gpu_tile(x, y, co, bx, by, bc, tx, ty, tc, 4, 4, 4)
            ;
          f[k].update()
            .gpu_tile(x, y, co, bx, by, bc, tx, ty, tc, 4, 4, 4)
            ;
          d(f[k]).compute_root();
        }

        for (std::string k : pool_keys) {
          f[k].compute_root()
            .gpu_tile(x, y, co, bx, by, bc, tx, ty, tc, 4, 4, 4);
          d(f[k]).compute_root();
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

        Var co_i("co_i");
        Var xi("xi");
        Var yi("yi");

        for (std::string k : conv_keys) {
          f[k]
            .compute_root()
            .fuse(n, co, co)
            .fuse(co, y, y)
            .parallel(y)
            .vectorize(x, v)
            ;
          RVar rx = f[k].rvars()[0];
          RVar ry = f[k].rvars()[1];
          RVar rz = f[k].rvars()[2];
          f[k]
            .update()
            .fuse(n, co, co)
            .fuse(co, y, y)
            .parallel(y)
            .vectorize(x, v)
            .unroll(rx)
            .unroll(ry)
            ;
        }

        for (std::string k : pool_keys) {
          f[k]
            .compute_root()
            .fuse(n, co, co)
            .fuse(co, y, y)
            .parallel(y)
            .vectorize(x, v)
            ;
        }

        f["fc6"].compute_root().parallel(co);
        f["fc6"].update().parallel(co);

        f["fc7"].compute_root().parallel(co);
        f["fc7"].update().parallel(co);

        f["output"].compute_root().parallel(co);
        f["output"].update().parallel(co);

        // Derivatives

        d(f["output"]).compute_root().parallel(co);
        d(f["fc7"]).compute_root().parallel(co);
        d(f["fc6"]).compute_root().parallel(co);
        // d(f["output"]).update().parallel(co);

        PrintFuncOptions opts;
        opts.depth = 12;
        print_func(d_fc_weights[2], opts);
        // autoschedule rfactors

      }
    }
  }

};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::VGGForwardBackwardGenerator, 
    vgg_forward_backward)

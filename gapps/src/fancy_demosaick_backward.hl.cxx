#include "algorithms/fancy_demosaick.h"
#include <vector>

#include "gradient_helpers.h"

namespace gradient_apps {

class FancyDemosaickBackwardGenerator : 
  public Generator<FancyDemosaickBackwardGenerator> {
public:
  Input<Buffer<float>>  cfa{"cfa", 3};
  Input<Func[n_w]> weights{"weights", Float(32), 1};
  Input<Func[n_w2]> weights2d{"weights2d", Float(32), 2};
  Input<Func[n_w3]> weights3d{"weights3d", Float(32), 3};
  Input<Func[n_w4]> weights4d{"weights4d", Float(32), 4};
  Input<Buffer<float>> d_output{"d_output", 4};

  Output<Func[n_w]> d_weights{"d_weights", Float(32), 1};
  Output<Func[n_w2]> d_weights2d{"d_weights2d", Float(32), 2};
  Output<Func[n_w3]> d_weights3d{"d_weights3d", Float(32), 3};
  Output<Func[n_w4]> d_weights4d{"d_weights4d", Float(32), 4};

  void generate() {
    std::map<std::string, Func> f = fancy_demosaick(
        cfa, weights, weights2d, weights3d, weights4d);
    Func f_output = f["output"];

    std::cout << "pre propag" << std::endl;
    Derivative d = propagate_adjoints(f_output, d_output,
        {
          {d_output.dim(0).min(), d_output.dim(0).max()},
          {d_output.dim(1).min(), d_output.dim(1).max()},
          {d_output.dim(2).min(), d_output.dim(2).max()},
          {d_output.dim(3).min(), d_output.dim(3).max()},
        });
    std::cout << "post propag" << std::endl;

    std::map<FuncKey, Func> adjoints = d.adjoints;
    std::cout << "adjoints\n"; 
    for(std::pair<FuncKey, Func> p : adjoints) {
      std::cout << p.second.name() <<"\n";
    }
    std::cout << "done with adjoints\n"; 

    PrintFuncOptions opts;
    // opts.depth = 1;
    // print_func(d_output, opts);

    std::set<std::string> skip = {};
    // skip.insert("N");
    // skip.insert("cd_0_d_def__");

    std::vector<Func> funcs;
    std::cout << "1d weights" << std::endl;
    for(int i = 0; i < n_w; ++i) {
      d_weights[i](x) = 0.0f;
      std::cout << "assigning " << weights[i].name() << "\n"; 
      // assign_gradient(d, weights[i], d_weights[i]);
      funcs.push_back(d_weights[i]);
    }
    std::cout << "2d weights" << std::endl;
    for(int i = 0; i < n_w2; ++i) {
      d_weights2d[i](x, y) = 0.0f;
      std::cout << "assigning " << weights2d[i].name() << "\n"; 
      // assign_gradient(d, weights2d[i], d_weights2d[i]);
      funcs.push_back(d_weights2d[i]);
    }
    std::cout << "3d weights" << std::endl;
    for(int i = 0; i < n_w3; ++i) {
      std::cout << "assigning " << weights3d[i].name() << "\n"; 
      assign_gradient(d, weights3d[i], d_weights3d[i]);
      funcs.push_back(d_weights3d[i]);
    }
    std::cout << "4d weights" << std::endl;
    for(int i = 0; i < n_w4; ++i) {
      std::cout << "assigning " << weights4d[i].name() << "\n"; 
      assign_gradient(d, weights4d[i], d_weights4d[i]);
      funcs.push_back(d_weights4d[i]);
    }
    std::cout << "derivatives set" << std::endl;

    SimpleAutoscheduleOptions options;
    options.gpu = get_target().has_gpu_feature();
    std::set<std::string> dont_inline = {};
    std::cout << "backward autoschedule" << std::endl;
    simple_autoschedule(funcs,
        {
        {"cfa.min.0", 0},
        {"cfa.min.1", 0},
        {"cfa.min.2", 0},
        {"cfa.extent.0", 128},
        {"cfa.extent.1", 128},
        {"cfa.extent.3", 1},
        {"weights3d_0.min.0", 0},
        {"weights3d_0.min.1", 0},
        {"weights3d_0.min.2", 0},
        {"weights3d_0.extent.0", 7},
        {"weights3d_0.extent.1", 7},
        {"weights3d_0.extent.2", 9},
        {"weights3d_1.min.0", 0},
        {"weights3d_1.min.1", 0},
        {"weights3d_1.min.2", 0},
        {"weights3d_1.extent.0", 7},
        {"weights3d_1.extent.1", 7},
        {"weights3d_1.extent.2", 9},
        {"weights3d_2.min.0", 0},
        {"weights3d_2.min.1", 0},
        {"weights3d_2.min.2", 0},
        {"weights3d_2.extent.0", 7},
        {"weights3d_2.extent.1", 7},
        {"weights3d_2.extent.2", 2*9},
        {"weights4d_0.min.0", 0},
        {"weights4d_0.min.1", 0},
        {"weights4d_0.min.2", 0},
        {"weights4d_0.min.3", 0},
        {"weights4d_0.extent.0", 7},
        {"weights4d_0.extent.1", 7},
        {"weights4d_0.extent.2", 4},
        {"weights4d_0.extent.3", 9},
        {"weights4d_1.min.0", 0},
        {"weights4d_1.min.1", 0},
        {"weights4d_1.min.2", 0},
        {"weights4d_1.min.3", 0},
        {"weights4d_1.extent.0", 7},
        {"weights4d_1.extent.1", 7},
        {"weights4d_1.extent.2", 4},
        {"weights4d_1.extent.3", 9},
        {"weights4d_2.min.0", 0},
        {"weights4d_2.min.1", 0},
        {"weights4d_2.min.2", 0},
        {"weights4d_2.min.3", 0},
        {"weights4d_2.extent.0", 7},
        {"weights4d_2.extent.1", 7},
        {"weights4d_2.extent.2", 4},
        {"weights4d_2.extent.3", 9},
        {"weights4d_3.min.0", 0},
        {"weights4d_3.min.1", 0},
        {"weights4d_3.min.2", 0},
        {"weights4d_3.min.3", 0},
        {"weights4d_3.extent.0", 7},
        {"weights4d_3.extent.1", 7},
        {"weights4d_3.extent.2", 4},
        {"weights4d_3.extent.3", 2*9},
        {"weights4d_4.min.0", 0},
        {"weights4d_4.min.1", 0},
        {"weights4d_4.min.2", 0},
        {"weights4d_4.min.3", 0},
        {"weights4d_4.extent.0", 7},
        {"weights4d_4.extent.1", 7},
        {"weights4d_4.extent.2", 4},
        {"weights4d_4.extent.3", 2*9},
        {"weights4d_5.min.0", 0},
        {"weights4d_5.min.1", 0},
        {"weights4d_5.min.2", 0},
        {"weights4d_5.min.3", 0},
        {"weights4d_5.extent.0", 7},
        {"weights4d_5.extent.1", 7},
        {"weights4d_5.extent.2", 4},
        {"weights4d_5.extent.3", 2*9},
        {"d_output.min.0", 0},
        {"d_output.min.1", 0},
        {"d_output.min.2", 0},
        {"d_output.min.3", 0},
        {"d_output.extent.0", 127},
        {"d_output.extent.1", 127},
        {"d_output.extent.2", 3},
        {"d_output.extent.3", 1},
        },
        {
          {{0, 6}},
          {{0, 6}, {0, 6}},
          {{0, 6}, {0, 6}, {0, 8}},
          {{0, 6}, {0, 6}, {0, 8}},
          {{0, 6}, {0, 6}, {0, 8}},
          {{0, 6}, {0, 6}, {0, 15}},
          {{0, 6}, {0, 6}, {0, 15}},
          {{0, 6}, {0, 6}, {0, 3}, {0, 8}},
          {{0, 6}, {0, 6}, {0, 3}, {0, 8}},
          {{0, 6}, {0, 6}, {0, 3}, {0, 8}},
          {{0, 6}, {0, 6}, {0, 3}, {0, 15}},
          {{0, 6}, {0, 6}, {0, 3}, {0, 15}},
          {{0, 6}, {0, 6}, {0, 3}, {0, 15}},
        },
        options,
        dont_inline, skip);
  }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::FancyDemosaickBackwardGenerator, 
    fancy_demosaick_backward)

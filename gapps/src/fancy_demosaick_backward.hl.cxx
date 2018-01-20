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
  Input<Buffer<float>> d_output{"d_output", 4};

  Output<Func[n_w]> d_weights{"d_weights", Float(32), 1};
  Output<Func[n_w2]> d_weights2d{"d_weights2d", Float(32), 2};

  void generate() {
    std::map<std::string, Func> f = fancy_demosaick(
        cfa, weights, weights2d);
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
    assert(adjoints.find(FuncKey{weights[0].name(), -1}) != adjoints.end());
    std::cout << "adjoints\n"; 
    for(std::pair<FuncKey, Func> p : adjoints) {
      std::cout << p.second.name() <<"\n";
    }
    std::cout << "done with adjoints\n"; 

    PrintFuncOptions opts;
    // opts.depth = 1;
    // print_func(d_output, opts);

    std::set<std::string> skip = {};
    skip.insert("N");

    std::vector<Func> funcs;
    for(int i = 0; i < n_w; ++i) {
      // d_weights[i](x) = 0.0f;
      std::cout << "assigning " << weights[i].name() << "\n"; 
      assign_gradient(d, weights[i], d_weights[i]);
      // d_weights[i] = d(weights[i]);
      funcs.push_back(d_weights[i]);
      // d_weights[i].gpu_tile(d_weights[i].args()[0], y, x, 3);
      // skip.insert(d_weights[i].name());
    }
    for(int i = 0; i < n_w2; ++i) {
      // d_weights2d[i](x, y) = 0.0f;
      std::cout << "assigning " << weights2d[i].name() << "\n"; 
      assign_gradient(d, weights2d[i], d_weights2d[i]);
      // d_weights2d[i] = d(weights2d[i]);
      funcs.push_back(d_weights2d[i]);
      // skip.insert(d_weights2d[i].name());
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
        {"cfa.extent.3", 4},
        {"weights_0.min.0", 0},
        {"weights_0.extent.0", 5},
        {"weights_1.min.0", 0},
        {"weights_1.extent.0", 5},
        {"weights_2.min.0", 0},
        {"weights_2.extent.0", 4},
        {"weights_3.min.0", 0},
        {"weights_3.extent.0", 5},
        {"weights_4.min.0", 0},
        {"weights_4.extent.0", 5},
        {"weights_5.min.0", 0},
        {"weights_5.extent.0", 4},
        {"weights2d_0.min.0", 0},
        {"weights2d_0.min.1", 0},
        {"weights2d_0.extent.0", 4},
        {"weights2d_0.extent.1", 4},
        {"weights2d_1.min.0", 0},
        {"weights2d_1.min.1", 0},
        {"weights2d_1.extent.0", 4},
        {"weights2d_1.extent.1", 4},
        {"weights2d_2.min.0", 0},
        {"weights2d_2.min.1", 0},
        {"weights2d_2.extent.0", 4},
        {"weights2d_2.extent.1", 4},
        {"weights2d_3.min.0", 0},
        {"weights2d_3.min.1", 0},
        {"weights2d_3.extent.0", 4},
        {"weights2d_3.extent.1", 4},
        {"d_output.min.0", 0},
        {"d_output.min.1", 0},
        {"d_output.min.2", 0},
        {"d_output.min.2", 0},
        {"d_output.extent.0", 127},
        {"d_output.extent.1", 127},
        {"d_output.extent.2", 3},
        {"d_output.extent.3", 4},
        },
        {
          {{0, 4}},
          {{0, 4}},
          {{0, 3}},
          {{0, 4}},
          {{0, 4}},
          {{0, 3}},
          {{0, 3}, {0, 3}},
          {{0, 3}, {0, 3}},
          {{0, 3}, {0, 3}},
          {{0, 3}, {0, 3}},
        },
        options,
        dont_inline, skip);
  }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::FancyDemosaickBackwardGenerator, 
    fancy_demosaick_backward)

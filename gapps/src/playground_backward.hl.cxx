#include "algorithms/playground.h"
#include <iostream>

using std::cout;
using std::endl;

namespace gradient_apps {

void apply_compute_root(Func F) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        cout << "computing root " << fit->first << endl;
        Func f(fit->second);
        f.compute_root();
    }
}

class PlaygroundBackwardGenerator : public Generator<PlaygroundBackwardGenerator> {
public:

    Input<Buffer<float>>  input1{"input1", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  input2{"input2", 4};       // x, y, channel, batch size
    Input<Buffer<float>> d_output{"d_output", 4};     // x, y, channel, batch size

    Output<Buffer<float>> d_input1{"d_input1", 4};   // same as input
    Output<Buffer<float>> d_input2{"d_input2", 4};   // same as input

    void generate() {
        std::map<std::string, Func> func_map = playground(
            input1, input2);

        Func f_output = func_map["output"];
        Func f_input1 = func_map["input1"];
        Func f_input2 = func_map["input2"];
        
        Derivative d = propagate_adjoints(f_output, d_output,
                                          {{d_output.dim(0).min(), d_output.dim(0).max()},
                                           {d_output.dim(1).min(), d_output.dim(1).max()},
                                           {d_output.dim(2).min(), d_output.dim(2).max()},
                                           {d_output.dim(3).min(), d_output.dim(3).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_input1.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_input2.name(), -1}) != adjoints.end());
        Func f_d_input1  = adjoints[FuncKey{f_input1.name(), -1}];
        Func f_d_input2  = adjoints[FuncKey{f_input2.name(), -1}];

        print_func(f_d_input1);
        print_func(f_d_input2);


        d_input1(x, y, ci, n) = f_d_input1(x, y, ci, n);
        d_input2(x, y, ci, n) = f_d_input2(x, y, ci, n);

        if(auto_schedule) {
          printf("Autoscheduling AHD demosaicking forward\n");
          int est_h = 512;
          int est_w = 512;
          input1.dim(0).set_bounds_estimate(0, est_w);
          input1.dim(1).set_bounds_estimate(0, est_h);
          input1.dim(2).set_bounds_estimate(0, 3);
          input1.dim(3).set_bounds_estimate(0, 16);

          input2.dim(0).set_bounds_estimate(0, est_w);
          input2.dim(1).set_bounds_estimate(0, est_h);
          input2.dim(2).set_bounds_estimate(0, 3);
          input2.dim(3).set_bounds_estimate(0, 16);

          d_output.dim(0).set_bounds_estimate(0, est_w);
          d_output.dim(1).set_bounds_estimate(0, est_h);
          d_output.dim(2).set_bounds_estimate(0, 3);
          d_output.dim(3).set_bounds_estimate(0, 16);

          d_input1.dim(0).set_bounds_estimate(0, est_w);
          d_input1.dim(1).set_bounds_estimate(0, est_h);
          d_input1.dim(2).set_bounds_estimate(0, 3);
          d_input1.dim(3).set_bounds_estimate(0, 16);

          d_input2.dim(0).set_bounds_estimate(0, est_w);
          d_input2.dim(1).set_bounds_estimate(0, est_h);
          d_input2.dim(2).set_bounds_estimate(0, 3);
          d_input2.dim(3).set_bounds_estimate(0, 16);
        } else {
          apply_compute_root(f_d_input1);
          apply_compute_root(f_d_input2);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::PlaygroundBackwardGenerator, playground_backward)

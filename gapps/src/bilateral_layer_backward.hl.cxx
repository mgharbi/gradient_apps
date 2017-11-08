#include "algorithms/bilateral_layer.h"
#include <iostream>

using std::cout;
using std::endl;

namespace gradient_apps {

std::map<std::string, Halide::Internal::Function> get_deps(Func F) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    cout << "Dependencies for " << F.name() << " " << endl;
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        cout << "  .Func " << fit->first << " " << endl;
        // Func f(fit->second);
        // f.compute_root();
    }
    return flist;
}

class BilateralLayerBackwardGenerator : public Generator<BilateralLayerBackwardGenerator> {
public:
    Input<int> sigma_x{"sigma_x"}; // block_size in x
    Input<int> sigma_y{"sigma_y"}; // block_size in y
    Input<int> sigma_z{"sigma_z"}; // number of guide discrete levels

    Input<Buffer<float>>  input{"input", 4};  // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};  // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};  // x, y, z offset, input channel, output channel

    Input<Buffer<float>>  d_output{"d_output", 4};   // x, y, out_channel, batch size

    Output<Buffer<float>> d_input{"d_input", 4};   // same as input
    Output<Buffer<float>> d_guide{"d_guide", 3};   // same as guide
    Output<Buffer<float>> d_filter{"d_filter", 5}; // same as filter

    void generate() {
        std::map<std::string, Func> func_map = bilateral_layer(
            input, guide, filter, sigma_x, sigma_y, sigma_z);

        Func f_output = func_map["output"];
        Func f_input = func_map["input"];
        Func f_guide = func_map["guide"];
        Func f_grid = func_map["grid"];
        Func f_filter = func_map["filter"];
        
        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_input.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_guide.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_filter.name(), -1}) != adjoints.end());

        Func f_d_input  = adjoints[FuncKey{f_input.name(), -1}];
        Func f_d_guide  = adjoints[FuncKey{f_guide.name(), -1}];
        Func f_d_filter = adjoints[FuncKey{f_filter.name(), -1}];

        d_input(x, y, ci, n) = f_d_input(x, y, ci, n);
        d_guide(x, y, n) = f_d_guide(x, y, n);
        d_filter(x, y, z, ci, co) = f_d_filter(x, y, z, ci, co);

        if(auto_schedule) {
          // Autoschedule
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

          d_output.dim(0).set_bounds_estimate(0, est_w);
          d_output.dim(1).set_bounds_estimate(0, est_h);
          d_output.dim(2).set_bounds_estimate(0, est_co);
          d_output.dim(3).set_bounds_estimate(0, est_bsize);

          d_input
            .estimate(x, 0, est_w)
            .estimate(y, 0, est_h)
            .estimate(ci, 0, est_ci)
            .estimate(n, 0, est_bsize)
            ;
          d_guide
            .estimate(x, 0, est_w)
            .estimate(y, 0, est_h)
            .estimate(n, 0, est_bsize)
            ;
          d_filter
            .estimate(x, 0, est_kw)
            .estimate(y, 0, est_kh)
            .estimate(z, 0, est_kd)
            .estimate(ci, 0, est_ci)
            .estimate(co, 0, est_co)
            ;

          printf("Autoscheduling bilateral_layer backward\n");
        } else {
          for(auto it=adjoints.begin(); it != adjoints.end(); ++it) {
            cout << "func " << it->first.first << " " << it->first.second << endl;
          }

          // TODO: 
          // - produce graph structure of the derivative computation
          // - give out handles to various wrapper locations to schedule the drv
          auto flist_guide = get_deps(d_guide);
          auto flist_filter = get_deps(d_filter);

          // Func conv_1_d_(flist_guide["conv_1_d__"]);
          // Func f_grid_1_d_(flist_guide["f_grid_1_d__"]);
          // Func f_grid_1_d_def_(flist_guide["f_grid_1_d_def__"]);
          // Func f_grid_2_d_(flist_guide["f_grid_2_d__"]);
          // Func f_grid_2_d_def_(flist_guide["f_grid_2_d_def__"]);

          // Forward schedule -------------------------------------------------
          func_map["grid"]
            .compute_root()
            .parallel(ci)
            .vectorize(x, 8)
            .update(0)
            .parallel(ci)
            .vectorize(x, 8)
            ;
          func_map["conv"]
            .compute_root()
            .parallel(co)
            .vectorize(x, 8)
            ;
          func_map["conv"]
            .update(0)
            .parallel(co)
            .vectorize(x, 8)
            ;
          func_map["output"]
            .compute_root()
            .parallel(co)
            .vectorize(x, 8)
            ;

          // print_func(d_input);
          // print_func(d_guide);
          // print_func(d_filter);

          // Backward schedule -------------------------------------------------


          // Func d_grid_init  = adjoints[FuncKey{func_map["grid"].name(), -1}];
          // Func d_grid0  = adjoints[FuncKey{func_map["grid"].name(), 0}];
          //
          // Func d_guide_init  = adjoints[FuncKey{func_map["guide"].name(), -1}];
          // Func d_input_init  = adjoints[FuncKey{func_map["input"].name(), -1}];

          // Func d_conv_def  = adjoints[FuncKey{func_map["conv"].name(), -1}];
          Func d_conv0  = adjoints[FuncKey{func_map["conv"].name(), 0}];
          // printf("d_conv: %s %s\n", d_conv_def.name().c_str(), d_conv0.name().c_str());
          d_conv0
            .compute_root()
            .parallel(co)
            .vectorize(x, 8)
            ;

          // Func d_grid_def  = adjoints[FuncKey{func_map["grid"].name(), -1}];
          Func d_grid0  = adjoints[FuncKey{func_map["grid"].name(), 0}];
          Func d_grid1  = adjoints[FuncKey{func_map["grid"].name(), 1}];
          // printf("d_grid: %s %s %s\n", d_grid_def.name().c_str(), d_grid0.name().c_str(), d_grid1.name().c_str());
          d_grid1
            .compute_root()
            .parallel(y, 4)
            .vectorize(x, 8)
            ;

          d_grid0
            .compute_root()
            .parallel(y, 4)
            .vectorize(x, 8)
            ;

          d_input
            .compute_root()
            .parallel(y, 4)
            .vectorize(x, 8)
            ;

          d_guide
            .compute_root()
            .parallel(y, 4)
            .vectorize(x, 8)
            ;

          d_filter
            .compute_root()
            .parallel(co)
            .vectorize(x, 8)
            ;

          // this is a scatter
          Func conv_1_d_def_(flist_guide["conv_1_d_def__"]);
          conv_1_d_def_.compute_at(d_conv0, co);

          Func d_guide_def(flist_guide["f_guide_0_d_def__"]);
          RDom rd = d.reductions[{d_guide_def.name(), 2}];
          Var rf("rf");
          cout << "rdom " << rd << endl;
          d_guide_def.update(2).rfactor(rd.y, rf).update().parallel(rf);

          // d_filter.print_loop_nest();
          // d_guide.print_loop_nest();
          // d_input.print_loop_nest();

        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralLayerBackwardGenerator, bilateral_layer_backward)

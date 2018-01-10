#include "algorithms/bilateral_layer.h"
#include "gradient_helpers.h"
#include <iostream>

using std::cout;
using std::endl;

namespace gradient_apps {

class BilateralLayerBackwardGenerator : public Generator<BilateralLayerBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 4};  // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};  // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};  // x, y, z offset, input channel, output channel
    Input<Buffer<float>>  d_output{"d_output", 4};   // x, y, out_channel, batch size

    Output<Buffer<float>> d_input{"d_input", 4};   // same as input
    Output<Buffer<float>> d_guide{"d_guide", 3};   // same as guide
    Output<Buffer<float>> d_filter{"d_filter", 5}; // same as filter

    void generate() {
      std::map<std::string, Func> func_map = bilateral_layer(
          input, guide, filter);

        Func f_output = func_map["output"];
        Func f_input = func_map["input"];
        Func f_guide = func_map["guide"];
        Func f_grid = func_map["grid"];
        Func f_filter = func_map["filter"];

        Derivative d = propagate_adjoints(
            f_output, 
            d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}}
             );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, f_input, d_input);
        assign_gradient(adjoints, f_guide, d_guide);
        assign_gradient(adjoints, f_filter, d_filter);

        SimpleAutoscheduleOptions options;
        options.gpu = get_target().has_gpu_feature();

        std::set<std::string> dont_inline = {};

        std::vector<Func> funcs{d_input, d_guide, d_filter};

        simple_autoschedule(funcs,
            {
              {"input.min.0", 0},
              {"input.min.1", 0},
              {"input.min.2", 0},
              {"input.min.3", 0},
              {"input.extent.0", 128},
              {"input.extent.1", 128},
              {"input.extent.2", 64},
              {"input.extent.3", 4},
              {"guide.min.0", 0},
              {"guide.min.1", 0},
              {"guide.min.2", 0},
              {"guide.extent.0", 128},
              {"guide.extent.1", 128},
              {"guide.extent.2", 4},
              {"filter.min.0", 0},
              {"filter.min.1", 0},
              {"filter.min.2", 0},
              {"filter.min.3", 0},
              {"filter.min.4", 0},
              {"filter.extent.0", 3},
              {"filter.extent.1", 3},
              {"filter.extent.2", 3},
              {"filter.extent.3", 64},
              {"filter.extent.4", 64},
              {"d_output.min.0", 0},
              {"d_output.min.1", 0},
              {"d_output.min.2", 0},
              {"d_output.min.3", 0},
              {"d_output.extent.0", 128},
              {"d_output.extent.1", 128},
              {"d_output.extent.2", 64},
              {"d_output.extent.3", 4},
            },
            { 
              {{0, 127}, {0, 127}, {0, 63}, {0, 3}}, //d_input
              {{0, 127}, {0, 127}, {0, 63}}, //d_guide
              {{0, 3}, {0, 3}, {0, 3}, {0, 63}, {0, 63}} //d_filter
            },
            options,
            dont_inline);

        // printf("Manually scheduling bilateral_layer backward\n");
        // // for(auto it=adjoints.begin(); it != adjoints.end(); ++it) {
        // //   cout << "func " << it->first.first << " " << it->first.second << endl;
        // // }
        //
        // printf("\nd_input deps:\n\n");
        // print_func(d_input);
        // printf("\nd_guide deps:\n\n");
        // print_func(d_guide);
        // printf("\nd_filter deps:\n\n");
        // print_func(d_filter);
        //
        // // Forward schedule -------------------------------------------------
        // func_map["grid"]
        //   .compute_root()
        //   .parallel(ci)
        //   .vectorize(x, 8)
        //   .update(0)
        //   .parallel(ci)
        //   .vectorize(x, 8)
        //   ;
        // func_map["grid"]
        //   .update(1)
        //   .parallel(ci)
        //   .vectorize(x, 8)
        //   ;
        // func_map["conv"]
        //   .compute_root()
        //   .parallel(n)
        //   .parallel(co)
        //   .parallel(z)
        //   .vectorize(x, 8)
        //   ;
        // func_map["conv"]
        //   .update(0)
        //   .parallel(n)
        //   .parallel(co)
        //   .parallel(z)
        //   .vectorize(x, 8)
        //   ;
        // func_map["output"]
        //   .compute_root()
        //   .parallel(n)
        //   .parallel(co)
        //   .parallel(y)
        //   .vectorize(x, 8)
        //   ;
        //
        // // Backward schedule -------------------------------------------------
        // auto flist_guide = get_deps(d_guide);
        // auto flist_filter = get_deps(d_filter);
        // auto flist_input = get_deps(d_input);
        //
        // // ----------------------------------------------------------------------
        // Func conv_1_d_def_(flist_guide["conv_1_d_def__"]);
        // conv_1_d_def_
        //   .compute_root()
        //   .parallel(y)
        //   .vectorize(x, 8)
        //   ;
        // // for(int i = 0; i<8; ++i) {
        // //   conv_1_d_def_
        // //     .update(i)
        // //     .parallel(co)
        // //     .parallel(n)
        // //     ;
        // // }
        //
        // Func conv_1_d_(flist_guide["conv_1_d__"]);
        // conv_1_d_
        //   .compute_root()
        //   .parallel(y)
        //   .vectorize(x, 8)
        //   ;
        //
        // Func f_grid_1_d_def_(flist_guide["f_grid_1_d_def__"]);
        // f_grid_1_d_def_
        //   .compute_root()
        //   .parallel(y)
        //   .vectorize(x, 8)
        //   ;
        // for (int i = 0; i < 3; ++i) {
        //   f_grid_1_d_def_
        //     .update(i)
        //     .parallel(y)
        //     .vectorize(x, 8)
        //     ;
        // }
        //
        //
        // Func f_grid_2_d_def_(flist_input["f_grid_2_d_def__"]);
        // std::vector<RVar> rvars = f_grid_2_d_def_.rvars(0);
        // for(RVar r: rvars) {
        //   cout << "rvar " << r.name() << "\n";
        // }
        // f_grid_2_d_def_
        //   .compute_root()
        //   .parallel(n)
        //   .parallel(ci)
        //   .parallel(z)
        //   .vectorize(x, 8)
        //   ;
        // f_grid_2_d_def_
        //   .update(0)
        //   .reorder(rvars[0], rvars[1], rvars[2], rvars[3])
        //   .parallel(n)
        //   .parallel(ci)
        //   .parallel(z)
        //   .vectorize(x, 8)
        //   ;
        //
        // Func input_0_d_def_(flist_input["f_input_0_d_def__"]);
        // input_0_d_def_
        //   .compute_root()
        //   .parallel(y, 4)
        //   .vectorize(x, 8)
        //   ;
        //
        // Func guide_0_d_def_(flist_guide["f_guide_0_d_def__"]);
        // guide_0_d_def_
        //   .compute_root()
        //   .parallel(y, 4)
        //   .vectorize(x, 8)
        //   ;
        // for(int i = 0; i<3; ++i) {
        //   guide_0_d_def_
        //     .update(i)
        //     .parallel(y, 4)
        //     .vectorize(x, 8)
        //     ;
        // }
        //
        // Func f_filter_0_d_def(flist_filter["f_filter_0_d_def__"]);
        // f_filter_0_d_def
        //   .compute_root()
        //   .parallel(co)
        //   .parallel(ci)
        //   .parallel(z)
        //   .parallel(y)
        //   .vectorize(x, 2)
        //   ;
        // // TODO: rfactor the update?
        // f_filter_0_d_def
        //   .update(0)
        //   .parallel(co)
        //   .parallel(ci)
        //   .parallel(z)
        //   .parallel(y)
        //   .vectorize(x, 2)
        //   ;
        //
        // // std::vector<RVar> rvars = f_filter_0_d_def.rvars(0);
        // // Var u;
        // // Func intermediate = f_filter_0_d_def.update(0).rfactor(rvars[0], u);
        // // intermediate.compute_root().update()
        // //   .parallel(u)
        // //   .parallel(co)
        // //   .parallel(ci)
        // //   .parallel(z)
        // //   .parallel(y);
        // // intermediate
        // //   .vectorize(x, 2);
        // // ----------------------------------------------------------------------
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralLayerBackwardGenerator, bilateral_layer_backward)

#include "algorithms/naive_demosaick.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class NaiveDemosaickBackwardGenerator 
  : public Generator<NaiveDemosaickBackwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 2};
    Input<Buffer<float>>  d_output{"d_output", 3};
    Output<Buffer<float>> d_mosaick{"d_mosaick", 2};

    void generate() {
        std::map<std::string, Func> func_map = naive_demosaick(mosaick);
        Func f_output = func_map["output"];
        Func f_mosaick = func_map["mosaick"];

        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_mosaick.name(), -1}) != adjoints.end());

        Func f_d_mosaick  = adjoints[FuncKey{f_mosaick.name(), -1}];

        d_mosaick(x, y) = f_d_mosaick(x, y);

        if(auto_schedule) {
        } else {
          Var xi("xi"), yi("yi"), xy("xy");
          auto deps = get_deps(d_mosaick);
          print_adjoints(adjoints);
          print_func(d_mosaick);

          Func d_red  = adjoints[FuncKey{"red", -1}];
          Func d_red_def = adjoints[FuncKey{"red_def__", -1}];
          Func d_green = adjoints[FuncKey{"green", -1}];
          Func d_green_def = adjoints[FuncKey{"green_def__", -1}];
          Func d_blue = adjoints[FuncKey{"blue", -1}];
          Func d_blue_def = adjoints[FuncKey{"blue_def__", -1}];
          Func d_chroma = adjoints[FuncKey{"chroma", -1}];
          Func d_chroma_def = adjoints[FuncKey{"chroma_def__", -1}];
          Func d_h_interp = adjoints[FuncKey{"h_interp", -1}];
          Func d_h_interp_def = adjoints[FuncKey{"h_interp_def__", -1}];
          Func d_v_interp = adjoints[FuncKey{"v_interp", -1}];
          Func d_v_interp_def = adjoints[FuncKey{"v_interp_def__", -1}];
          Func d_q_interp = adjoints[FuncKey{"q_interp", -1}];
          Func d_q_interp_def = adjoints[FuncKey{"q_interp_def__", -1}];
          Func d_mosaick_def = adjoints[FuncKey{"f_mosaick_def__", -1}];
          Func d_igreen = adjoints[FuncKey{"interpolated_green", -1}];
          Func d_igreen_def = adjoints[FuncKey{"interpolated_green_def__", -1}];

          d_blue_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            ;
          d_blue_def
            .update()
            .vectorize(x, 8)
            ;
          d_red_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            ;
          d_red_def
            .update()
            .vectorize(x, 8)
            ;

          d_v_interp_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            .update(0)
            .vectorize(x, 8)
            ;
          d_v_interp_def
            .update(1)
            .vectorize(x, 8)
            ;

          d_h_interp_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            .update(0)
            .vectorize(x, 8)
            ;
          d_h_interp_def
            .update(1)
            .vectorize(x, 8)
            ;

          d_q_interp_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            .update(0)
            .vectorize(x, 8)
            ;
          d_q_interp_def
            .update(1)
            .vectorize(x, 8)
            ;

          d_chroma_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            ;
          for (int i = 0; i < 8; ++i) {
            d_chroma_def
              .update(i)
              .vectorize(x, 8)
              ;
          }

          d_green_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            ;
          for (int i = 0; i < 5; ++i) {
            d_green_def
              .update(i)
              .vectorize(x, 8)
              ;
          }

          d_igreen_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            ;
          d_igreen_def
            .update()
            .vectorize(x, 8)
            ;

          d_mosaick_def
            .compute_at(d_mosaick, xy)
            .vectorize(x, 8)
            ;
          for (int i = 0; i < 5; ++i) {
            d_mosaick_def
              .update(i)
              .vectorize(x, 8)
              ;
          }
          d_mosaick
            .tile(x, y, xi, yi, 16, 16)
            .fuse(x, y, xy)
            .parallel(xy)
            .vectorize(xi, 8)
            ;
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::NaiveDemosaickBackwardGenerator, naive_demosaick_backward)

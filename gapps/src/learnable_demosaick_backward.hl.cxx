#include "algorithms/learnable_demosaick.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class LearnableDemosaickBackwardGenerator
  : public Generator<LearnableDemosaickBackwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 3};
    Input<Buffer<float>>  sel_filts{"selection_filters", 3};
    Input<Buffer<float>>  green_filts{"green_filters", 3};
    Input<Buffer<float>>  d_output{"d_output", 4};

    Output<Buffer<float>> d_mosaick{"d_mosaick", 3};
    Output<Buffer<float>> d_sel_filts{"d_selection_filters", 3};
    Output<Buffer<float>> d_green_filts{"d_green_filters", 3};

    void generate() {
        std::map<std::string, Func> func_map = learnable_demosaick(mosaick, sel_filts, green_filts);
        Func f_output = func_map["output"];
        Func f_mosaick = func_map["mosaick"];
        Func f_sel_filts = func_map["selection_filters"];
        Func f_green_filts = func_map["green_filters"];

        Derivative d = propagate_adjoints(
            f_output, d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}
             });
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assert(adjoints.find(FuncKey{f_mosaick.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_sel_filts.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_green_filts.name(), -1}) != adjoints.end());

        Func f_d_mosaick  = adjoints[FuncKey{f_mosaick.name(), -1}];
        Func f_d_sel_filts  = adjoints[FuncKey{f_sel_filts.name(), -1}];
        Func f_d_green_filts  = adjoints[FuncKey{f_green_filts.name(), -1}];

        d_mosaick(x, y, n) = f_d_mosaick(x, y, n);
        d_sel_filts(x, y, n) = f_d_sel_filts(x, y, n);
        d_green_filts(x, y, n) = f_d_green_filts(x, y, n);

        // TODO: remove the need to wrap Halide::Interal::function
        std::map<std::string, Halide::Internal::Function> l = get_deps({d_sel_filts, d_green_filts, d_mosaick});

        Var xi("xi"), yi("yi"), xy("xy"), xyn("xyn");
        Var xo("xo"), yo("yo"), xyk("xyk"), xykn("xykn");
        Var xiyi("xiyi");

        for(auto m : l) {
          cerr << m.first << " " << Func(m.second).num_update_definitions() <<"\n";
          // Func(m.second).compute_root();
        }
        cerr << "\n";

        // print_deps(d_sel_filts);
        // print_deps(d_green_filts);
        if (get_target().has_gpu_feature()) {
          cerr << "gpu schedule\n";
          int ts = 8;

          Func(l["f_output_0_d_def__"])
            .compute_root()
            ;
          Func(l["blue_0_d_def__"])
            .compute_root()
            ;
          Func(l["red_0_d_def__"])
            .compute_root()
            ;
          Func(l["h_interp_0_d_def__"])
            .compute_root()
            ;
          Func(l["q_interp_0_d_def__"])
            .compute_root()
            ;
          Func(l["v_interp_0_d_def__"])
            .compute_root()
            ;
          Func(l["chroma_0_d_def__"])
            .compute_root()
            ;
          Func(l["green_0_d_def__"])
            .compute_root()
            ;
          Func(l["interpolated_green_1_d_def__"])
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
          Func(l["selection"])
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
          Func(l["normalizer"])
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
          Func(l["interp_g"])
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
          // Func(l["weights"])
          //   .compute_root()
          //   ;
          Func(l["interp_g_1_d_def__"])
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
          Func f_expsel(l["exp_selection_0_d_def__"]);
          f_expsel
            .compute_root()
            .reorder(k, x, y, n)
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
          for(int i = 0; i < f_expsel.num_update_definitions(); ++i) {
            f_expsel.update(i)
              .reorder(k, x, y, n)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;
          }
          Func(l["selection_1_d_def__"])
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
          Func(l["weights_0_d_def__"])
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
          Func(l["normalizer_1_d_def__"])
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;

          Func fmos(l["f_mosaick_0_d_def__"]);
          fmos
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
          for(int i = 0; i < fmos.num_update_definitions(); ++i) {
            fmos.update(i)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;
          }

          Func fsel(l["f_sel_filts_0_d_def__"]);
          std::vector<RVar> fsel_v = fsel.rvars(0);
          fsel.compute_root();

          // Resplit the reduction domain
          // RVar r("r");
          // RVar r1("r1");
          // RVar r2("r2");
          // RVar r3("r3");
          // int factor = 64;
          // fsel.update()
          //   .fuse(fsel_v[0], fsel_v[1], r)
          //   .fuse(fsel_v[2], r, r)
          //   .split(r, r1, r2, factor*factor)
          //   .split(r2, r2, r3, factor)
          //   ;

          Var r1_i("r1_i");
          Func fsel_1 = fsel.update()
            .rfactor(fsel_v[2], r1_i);

          Var r2_i("r2_i");
          Func fsel_2 = fsel_1.update()
            .rfactor(fsel_v[1], r2_i);
          fsel_2
            .compute_at(fsel_1, r1_i)
            .reorder(r2_i, x, y, k)
            // .gpu_tile(r2_i, xi, ts)
            .update()
            .gpu_threads(r2_i)
            // .gpu_tile(r2_i, xi, ts)
            ;
          
          fsel_1
            .compute_at(fsel, x)
            .reorder(r1_i, x, y, k)
            // .gpu_tile(r1_i, yi, ts)
            .update()
            .gpu_tile(r1_i, yi, ts)
            ;

          fsel.print_loop_nest();
          print_func(fsel, true, 3);

          Func(l["f_green_filts_0_d_def__"])
            .compute_root()
            ;
        } else {
          // compute_all_root(d_sel_filts);
          cerr << "cpu schedule\n";
          for(auto m : l) {
            Func(m.second).compute_root();
          }
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::LearnableDemosaickBackwardGenerator, learnable_demosaick_backward)

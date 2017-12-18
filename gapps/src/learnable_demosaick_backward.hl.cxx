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

        // TODO: remove the need to wrap Halide::Internal::function
        std::map<std::string, Halide::Internal::Function> l = get_deps({d_sel_filts, d_green_filts, d_mosaick});

        Var xi("xi"), yi("yi"), xy("xy"), xyn("xyn"), xynk("xynk");
        Var xo("xo"), yo("yo"), xyk("xyk"), xykn("xykn");
        Var xiyi("xiyi");

        for(auto m : l) {
          cerr << m.first << " " << Func(m.second).num_update_definitions() <<"\n";
        }
        cerr << "\n";

        if (get_target().has_gpu_feature()) {
          cerr << "gpu schedule\n";
          int ts = 8;

          Func(l["blue_0_d_def__"])
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
          Func(l["red_0_d_def__"])
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
          Func d_hinterp(l["h_interp_0_d_def__"]);
          d_hinterp
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts);
          for(int i = 0; i < d_hinterp.num_update_definitions(); ++i) {
            d_hinterp
              .update(i)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;
          }
          Func d_vinterp(l["v_interp_0_d_def__"]);
          d_vinterp
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts);
          for(int i = 0; i < d_vinterp.num_update_definitions(); ++i) {
            d_vinterp
              .update(i)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;
          }
          Func d_qinterp(l["q_interp_0_d_def__"]);
          d_qinterp
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts);
          for(int i = 0; i < d_qinterp.num_update_definitions(); ++i) {
            d_qinterp
              .update(i)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;
          }
          Func d_chroma(l["chroma_0_d_def__"]);
          d_chroma
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
          for(int i = 0; i < d_chroma.num_update_definitions(); ++i) {
            d_chroma
              .update(i)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts)
              ;
          }
          Func d_green(l["green_0_d_def__"]);
          RVar d_green_v = d_green.rvars(0)[0];
          d_green
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts*ts)
            ;
          for(int i = 0; i < d_green.num_update_definitions(); ++i) {
            if ( i == 0) {
              d_green.update(i).unroll(d_green_v, 3);
            }
            d_green
              .update(i)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .gpu_tile(xyn, xi, ts*ts)
              ;
          }
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
          Func(l["sel_max"])
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
          Func(l["abs_selection"])
            .compute_root()
            .reorder(k, x, y, n)
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
          Func(l["weights"])
            .compute_root()
            .reorder(k, x, y, n)
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
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
          Func d_weights(l["weights_0_d_def__"]);
          d_weights
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .fuse(xyn, k, xynk)
            .gpu_tile(xynk, xi, ts*ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .fuse(xyn, k, xynk)
            .gpu_tile(xynk, xi, ts*ts)
            ;
          Func f_expsel(l["exp_selection_0_d_def__"]);
          f_expsel
            .compute_root()
            .reorder(k, x, y, n)
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
          Func(l["sel_max_1_d_def__"])
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;
          Func(l["abs_selection_0_d_def__"])
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            .update()
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
          } // TODO: last two updates need rfactors
          d_mosaick
            .compute_root()
            .fuse(x, y, xy)
            .fuse(xy, n, xyn)
            .gpu_tile(xyn, xi, ts)
            ;

          // Hierarchical reduction --------------------
          Var r1_i("r1_i");
          Var r2_i("r2_i");
          Func fsel(l["f_sel_filts_0_d_def__"]);
          std::vector<RVar> fsel_v = fsel.rvars(0);
          fsel
            .compute_root();

          Func fsel_1 = fsel.update()
            .rfactor(fsel_v[2], r1_i);
          Func fsel_2 = fsel_1.update()
            .rfactor(fsel_v[1], r2_i)
            ;

          fsel
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            ;

          fsel_2
            .compute_at(fsel_1, r1_i)
            .reorder(r2_i, x, y, k)
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .gpu_threads(r2_i)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .gpu_threads(r2_i)
            ;

          fsel_1
            .compute_at(fsel, xyn)
            .reorder(r1_i, x, y, k)
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .gpu_tile(r1_i, yi, ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .gpu_tile(r1_i, yi, ts)
            ;
          fsel.print_loop_nest();
          print_func(fsel, true, 3);
          // -------------------------------------------

          // Hierarchical reduction --------------------
          Func fgreen(l["f_green_filts_0_d_def__"]);
          std::vector<RVar> fgreen_v = fgreen.rvars(0);
          fgreen
            .compute_root();

          Func fgreen_1 = fgreen.update()
            .rfactor(fgreen_v[2], r1_i);
          Func fgreen_2 = fgreen_1.update()
            .rfactor(fgreen_v[1], r2_i)
            ;

          fgreen
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            ;

          fgreen_2
            .compute_at(fgreen_1, r1_i)
            .reorder(r2_i, x, y, k)
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .gpu_threads(r2_i)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .gpu_threads(r2_i)
            ;

          fgreen_1
            .compute_at(fgreen, xyn)
            .reorder(r1_i, x, y, k)
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .gpu_tile(r1_i, yi, ts)
            .update()
            .fuse(x, y, xy)
            .fuse(xy, k, xyn)
            .gpu_tile(r1_i, yi, ts)
            ;
          // -------------------------------------------
          
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

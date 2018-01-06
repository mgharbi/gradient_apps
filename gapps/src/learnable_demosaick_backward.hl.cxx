#include "algorithms/learnable_demosaick.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class LearnableDemosaickBackwardGenerator
  : public Generator<LearnableDemosaickBackwardGenerator> {
public:
    Input<Buffer<float>>  mosaick{"mosaick", 3};
    Input<Buffer<float>>  sel_filts{"selection_filters", 3};
    Input<Buffer<float>>  green_filts{"green_filters", 3};
    Input<Buffer<float>>  h_chroma_filter{"h_chroma_filter", 2};
    Input<Buffer<float>>  v_chroma_filter{"v_chroma_filter", 2};
    Input<Buffer<float>>  q_chroma_filter{"q_chroma_filter", 2};
    Input<Buffer<float>>  d_output{"d_output", 4};

    Output<Buffer<float>> d_mosaick{"d_mosaick", 3};
    Output<Buffer<float>> d_sel_filts{"d_selection_filters", 3};
    Output<Buffer<float>> d_green_filts{"d_green_filters", 3};
    Output<Buffer<float>> d_h_chroma_filter{"d_h_chroma_filter", 2};
    Output<Buffer<float>> d_v_chroma_filter{"d_v_chroma_filter", 2};
    Output<Buffer<float>> d_q_chroma_filter{"d_q_chroma_filter", 2};

    void generate() {
        std::map<std::string, Func> func_map = learnable_demosaick(
            mosaick, sel_filts, green_filts,
            h_chroma_filter, v_chroma_filter, q_chroma_filter);
        Func f_output = func_map["output"];
        Func f_mosaick = func_map["mosaick"];
        Func f_sel_filts = func_map["selection_filters"];
        Func f_green_filts = func_map["green_filters"];
        Func f_h_chroma_filter = func_map["h_chroma_filter"];
        Func f_v_chroma_filter = func_map["v_chroma_filter"];
        Func f_q_chroma_filter = func_map["q_chroma_filter"];

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
        assert(adjoints.find(FuncKey{f_h_chroma_filter.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_v_chroma_filter.name(), -1}) != adjoints.end());
        assert(adjoints.find(FuncKey{f_q_chroma_filter.name(), -1}) != adjoints.end());

        Func f_d_mosaick  = adjoints[FuncKey{f_mosaick.name(), -1}];
        Func f_d_sel_filts  = adjoints[FuncKey{f_sel_filts.name(), -1}];
        Func f_d_green_filts  = adjoints[FuncKey{f_green_filts.name(), -1}];
        Func f_d_h_chroma_filter  = adjoints[FuncKey{f_h_chroma_filter.name(), -1}];
        Func f_d_v_chroma_filter  = adjoints[FuncKey{f_v_chroma_filter.name(), -1}];
        Func f_d_q_chroma_filter  = adjoints[FuncKey{f_q_chroma_filter.name(), -1}];

        d_mosaick(x, y, n) = f_d_mosaick(x, y, n);
        d_sel_filts(x, y, n) = f_d_sel_filts(x, y, n);
        d_green_filts(x, y, n) = f_d_green_filts(x, y, n);
        d_h_chroma_filter(x, y) = f_d_h_chroma_filter(x, y);
        d_v_chroma_filter(x, y) = f_d_v_chroma_filter(x, y);
        d_q_chroma_filter(x, y) = f_d_q_chroma_filter(x, y);

        SimpleAutoscheduleOptions options;
        options.gpu = get_target().has_gpu_feature();

        Func d_mosaick_func = d_mosaick;
        Func d_sel_filts_func = d_sel_filts;
        Func d_green_filts_func = d_green_filts;
        Func d_h_chroma_filter_func = d_h_chroma_filter;
        Func d_v_chroma_filter_func = d_v_chroma_filter;
        Func d_q_chroma_filter_func = d_q_chroma_filter;
        std::set<std::string> dont_inline = {};

        std::vector<Func> funcs{d_mosaick, d_sel_filts, d_green_filts,
          d_h_chroma_filter, d_v_chroma_filter, d_q_chroma_filter};

        simple_autoschedule(funcs,
            {
              {"mosaick.min.0", 0},
              {"mosaick.min.1", 0},
              {"mosaick.min.2", 0},
              {"mosaick.extent.0", 64},
              {"mosaick.extent.1", 64},
              {"mosaick.extent.2", 16},
              {"selection_filters.min.0", 0},
              {"selection_filters.min.1", 0},
              {"selection_filters.min.2", 0},
              {"selection_filters.extent.0", 8},
              {"selection_filters.extent.1", 5},
              {"selection_filters.extent.2", 5},
              {"green_filters.min.0", 0},
              {"green_filters.min.1", 0},
              {"green_filters.min.2", 0},
              {"green_filters.extent.0", 8},
              {"green_filters.extent.1", 5},
              {"green_filters.extent.2", 5},
              {"h_chroma_filter.min.0", 0},
              {"h_chroma_filter.min.1", 0},
              {"h_chroma_filter.extent.0", 5},
              {"h_chroma_filter.extent.1", 5},
              {"v_chroma_filter.min.0", 0},
              {"v_chroma_filter.min.1", 0},
              {"v_chroma_filter.extent.0", 5},
              {"v_chroma_filter.extent.1", 5},
              {"q_chroma_filter.min.0", 0},
              {"q_chroma_filter.min.1", 0},
              {"q_chroma_filter.extent.0", 5},
              {"q_chroma_filter.extent.1", 5},
              {"d_output.min.0", 0},
              {"d_output.min.1", 0},
              {"d_output.min.2", 0},
              {"d_output.min.3", 0},
              {"d_output.extent.0", 64},
              {"d_output.extent.1", 64},
              {"d_output.extent.2", 3},
              {"d_output.extent.3", 16}
            },
            {
              {{0, 63}, {0, 63}, {0, 15}},  // d_mosaick
              {{0, 4}, {0, 4}, {0, 7}},  // d_sel_filts
              {{0, 4}, {0, 4}, {0, 7}},  // d_green_filts
              {{0, 4}, {0, 4}},  // d_h_chroma
              {{0, 4}, {0, 4}},  // d_v_chroma
              {{0, 4}, {0, 4}}  // d_q_chroma
            },
            options,
            dont_inline);

        // // TODO: remove the need to wrap Halide::Internal::function
        // std::map<std::string, Halide::Internal::Function> l = get_deps({d_sel_filts, d_green_filts, d_mosaick});
        //
        // Var xi("xi"), yi("yi"), xy("xy"), xyn("xyn"), xynk("xynk");
        // Var xo("xo"), yo("yo"), xyk("xyk"), xykn("xykn");
        // Var xiyi("xiyi");
        //
        // for(auto m : l) {
        //   cerr << m.first << " " << Func(m.second).num_update_definitions() <<"\n";
        // }
        // cerr << "\n";
        //
        // if (get_target().has_gpu_feature()) {
        //   cerr << "gpu schedule\n";
        //   int ts = 8;
        //
        //   Func(l["blue_0_d_def__"])
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["red_0_d_def__"])
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func d_hinterp(l["h_interp_0_d_def__"]);
        //   d_hinterp
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts);
        //   for(int i = 0; i < d_hinterp.num_update_definitions(); ++i) {
        //     d_hinterp
        //       .update(i)
        //       .fuse(x, y, xy)
        //       .fuse(xy, n, xyn)
        //       .gpu_tile(xyn, xi, ts)
        //       ;
        //   }
        //   Func d_vinterp(l["v_interp_0_d_def__"]);
        //   d_vinterp
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts);
        //   for(int i = 0; i < d_vinterp.num_update_definitions(); ++i) {
        //     d_vinterp
        //       .update(i)
        //       .fuse(x, y, xy)
        //       .fuse(xy, n, xyn)
        //       .gpu_tile(xyn, xi, ts)
        //       ;
        //   }
        //   Func d_qinterp(l["q_interp_0_d_def__"]);
        //   d_qinterp
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts);
        //   for(int i = 0; i < d_qinterp.num_update_definitions(); ++i) {
        //     d_qinterp
        //       .update(i)
        //       .fuse(x, y, xy)
        //       .fuse(xy, n, xyn)
        //       .gpu_tile(xyn, xi, ts)
        //       ;
        //   }
        //   Func d_chroma(l["chroma_0_d_def__"]);
        //   d_chroma
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   for(int i = 0; i < d_chroma.num_update_definitions(); ++i) {
        //     d_chroma
        //       .update(i)
        //       .fuse(x, y, xy)
        //       .fuse(xy, n, xyn)
        //       .gpu_tile(xyn, xi, ts)
        //       ;
        //   }
        //   Func d_green(l["green_0_d_def__"]);
        //   RVar d_green_v = d_green.rvars(0)[0];
        //   d_green
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts*ts)
        //     ;
        //   for(int i = 0; i < d_green.num_update_definitions(); ++i) {
        //     if ( i == 0) {
        //       d_green.update(i).unroll(d_green_v, 3);
        //     }
        //     d_green
        //       .update(i)
        //       .fuse(x, y, xy)
        //       .fuse(xy, n, xyn)
        //       .gpu_tile(xyn, xi, ts*ts)
        //       ;
        //   }
        //   Func(l["interpolated_green_1_d_def__"])
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["sel_max"])
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["selection"])
        //     .compute_root()
        //     .reorder(k, x, y, n)
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["abs_selection"])
        //     .compute_root()
        //     .reorder(k, x, y, n)
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["normalizer"])
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["interp_g"])
        //     .compute_root()
        //     .reorder(k, x, y, n)
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["weights"])
        //     .compute_root()
        //     .reorder(k, x, y, n)
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["interp_g_1_d_def__"])
        //     .compute_root()
        //     .reorder(k, x, y, n)
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func d_weights(l["weights_0_d_def__"]);
        //   d_weights
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .fuse(xyn, k, xynk)
        //     .gpu_tile(xynk, xi, ts*ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .fuse(xyn, k, xynk)
        //     .gpu_tile(xynk, xi, ts*ts)
        //     ;
        //   Func f_expsel(l["exp_selection_0_d_def__"]);
        //   f_expsel
        //     .compute_root()
        //     .reorder(k, x, y, n)
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["normalizer_1_d_def__"])
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["sel_max_1_d_def__"])
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   Func(l["abs_selection_0_d_def__"])
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   for(int i = 0; i < f_expsel.num_update_definitions(); ++i) {
        //     f_expsel.update(i)
        //       .reorder(k, x, y, n)
        //       .fuse(x, y, xy)
        //       .fuse(xy, n, xyn)
        //       .gpu_tile(xyn, xi, ts)
        //       ;
        //   }
        //   Func(l["selection_1_d_def__"])
        //     .compute_root()
        //     .reorder(k, x, y, n)
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //
        //   Func fmos(l["f_mosaick_0_d_def__"]);
        //   fmos
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //   for(int i = 0; i < fmos.num_update_definitions(); ++i) {
        //     fmos.update(i)
        //       .fuse(x, y, xy)
        //       .fuse(xy, n, xyn)
        //       .gpu_tile(xyn, xi, ts)
        //       ;
        //   } // TODO: last two updates need rfactors
        //   d_mosaick
        //     .compute_root()
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
        //
        //   // Hierarchical reduction --------------------
        //   Var r1_i("r1_i");
        //   Var r2_i("r2_i");
        //   Func fsel(l["f_sel_filts_0_d_def__"]);
        //   std::vector<RVar> fsel_v = fsel.rvars(0);
        //   fsel
        //     .compute_root();
        //
        //   Func fsel_1 = fsel.update()
        //     .rfactor(fsel_v[2], r1_i);
        //   Func fsel_2 = fsel_1.update()
        //     .rfactor(fsel_v[1], r2_i)
        //     ;
        //
        //   fsel
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     ;
        //
        //   fsel_2
        //     .compute_at(fsel_1, r1_i)
        //     .reorder(r2_i, x, y, k)
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .gpu_threads(r2_i)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .gpu_threads(r2_i)
        //     ;
        //
        //   fsel_1
        //     .compute_at(fsel, xyn)
        //     .reorder(r1_i, x, y, k)
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .gpu_tile(r1_i, yi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .gpu_tile(r1_i, yi, ts)
        //     ;
        //   fsel.print_loop_nest();
        //   print_func(fsel, true, 3);
        //   // -------------------------------------------
        //
        //   // Hierarchical reduction --------------------
        //   Func fgreen(l["f_green_filts_0_d_def__"]);
        //   std::vector<RVar> fgreen_v = fgreen.rvars(0);
        //   fgreen
        //     .compute_root();
        //
        //   Func fgreen_1 = fgreen.update()
        //     .rfactor(fgreen_v[2], r1_i);
        //   Func fgreen_2 = fgreen_1.update()
        //     .rfactor(fgreen_v[1], r2_i)
        //     ;
        //
        //   fgreen
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     ;
        //
        //   fgreen_2
        //     .compute_at(fgreen_1, r1_i)
        //     .reorder(r2_i, x, y, k)
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .gpu_threads(r2_i)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .gpu_threads(r2_i)
        //     ;
        //
        //   fgreen_1
        //     .compute_at(fgreen, xyn)
        //     .reorder(r1_i, x, y, k)
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .gpu_tile(r1_i, yi, ts)
        //     .update()
        //     .fuse(x, y, xy)
        //     .fuse(xy, k, xyn)
        //     .gpu_tile(r1_i, yi, ts)
        //     ;
        //   // -------------------------------------------
        //   
        // } else {
        //   // compute_all_root(d_sel_filts);
        //   cerr << "cpu schedule\n";
        //   for(auto m : l) {
        //     Func(m.second).compute_root();
        //   }
        // }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::LearnableDemosaickBackwardGenerator, learnable_demosaick_backward)

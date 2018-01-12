#ifndef GRADIENT_HELPERS_H_FSA3FYYR
#define GRADIENT_HELPERS_H_FSA3FYYR

#include <iostream>
#include <map>
#include <string>
#include "Halide.h"

using std::cerr;
using std::endl;

using namespace Halide;

void print_deps(Func F) {
    std::map<std::string, Halide::Internal::Function> flist =
        Halide::Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    cerr << "Dependencies for " << F.name() << " " << endl;
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        cerr << "  .Func " << fit->first << " " << "\n";
    }
}

std::map<std::string, Halide::Internal::Function> get_deps(Func F) {
    std::map<std::string, Halide::Internal::Function> flist =
        Halide::Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    // cerr << "Dependencies for " << F.name() << " " << endl;
    // for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
    //     cerr << "  .Func " << fit->first << " " << "\n";
    // }
    return flist;
}

std::map<std::string, Halide::Internal::Function> get_deps(std::vector<Func> v) {
  std::map<std::string, Halide::Internal::Function> flist;
  for(Func f : v) {
    std::map<std::string, Halide::Internal::Function> curr = get_deps(f);
    flist.insert(curr.begin(), curr.end());
  }
  return flist;
}

void compute_all_root(Func F) {
    std::map<std::string, Halide::Internal::Function> flist =
        Halide::Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        std::vector<Var> args = f.args();
        // cerr << "Func " << f.name() <<" with " << args.size() << " args\n" ;

        f.compute_root();

        // // Vectorize inner most
        // if(args.size() > 0) {
        //   cerr << "arg0 " << args[0].name() << "\n";
        //   Var inner_most = args[0];
        //   // f.vectorize(inner_most, 4);
        // }

    //     // Parallel on all other dims
    //     if(args.size() > 1) {
    //       Var new_var = args[1];
    //         // cerr << "arg " << 1 << " " << args[1].name() << "\n";
    //       for(int i = 2; i < args.size(); ++i) {
    //         // cerr << "arg " << i << " " << args[i].name() << "\n";
    //         f.fuse(new_var, args[i], new_var);
    //       }
    //       f.parallel(new_var);
    //     }
    }
}

void compute_all_at(Func F, Func at_target, Var loc) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(F.function());
    // flist.insert(std::make_pair(F.name(), F.function()));
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        if (f.name() == at_target.name()) {
          continue;
        }
        f.in(at_target).compute_at(at_target, loc);
    }
}

void print_adjoints(std::map<FuncKey, Func> adjoints) {
  for(auto f : adjoints) {
    cerr << f.first.first << " " << f.first.second << "\n";
  }
}

template <typename Input, typename Output>
void assign_gradient(std::map<FuncKey, Func> &adjoints,
                     const Input &func,
                     Output &output) {
    if (adjoints.find(FuncKey{func.name(), -1}) != adjoints.end()) {
        output(_) = adjoints[FuncKey{func.name(), -1}](_);
    } else {
        std::cerr << "func.name():" << func.name() << std::endl;
        assert(false);
        output(_) = 0.f;
    }
}

Func get_func(const std::map<std::string, Halide::Internal::Function> &func_map,
              const std::string &name) {
    auto it = func_map.find(name);
    if (it == func_map.end()) {
        std::cerr << "Can't find function " << name << " in func_map" << std::endl;
        assert(false);
    }
    return Func(it->second);
}

std::pair<Func, Func> select_repeat_edge(Func input, Expr width, Expr height) {
    std::vector<Var> args = input.args();
    assert(args.size() >= 2);
    Func repeat_edge(input.name() + std::string("repeat_edge"));
    std::vector<Expr> exprs;
    exprs.push_back(clamp(likely(args[0]), 0, width-1));
    exprs.push_back(clamp(likely(args[1]), 0, height-1));
    for (int i = 2; i < (int)args.size(); i++) {
        exprs.push_back(args[i]);
    }
    repeat_edge(args) = input(exprs);
    Func selected(input.name() + std::string("selected"));
    std::vector<Expr> arg_exprs;
    for (int i = 0; i < (int)args.size(); i++) {
        arg_exprs.push_back(args[i]);
    }
    exprs = arg_exprs;
    exprs[0] = Expr(0);
    selected(args) = select(args[0] >= 0, repeat_edge(arg_exprs), repeat_edge(exprs));
    exprs[0] = width - 1;
    selected(args) = select(args[0] < width, selected(arg_exprs), repeat_edge(exprs));
    exprs[0] = args[0];
    exprs[1] = Expr(0);
    selected(args) = select(args[1] >= 0, selected(arg_exprs), repeat_edge(exprs));
    exprs[1] = height - 1;
    selected(args) = select(args[1] < height, selected(arg_exprs), repeat_edge(exprs));
    return {repeat_edge, selected};
}

#endif /* end of include guard: GRADIENT_HELPERS_H_FSA3FYYR */

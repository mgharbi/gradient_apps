#ifndef GRADIENT_HELPERS_H_FSA3FYYR
#define GRADIENT_HELPERS_H_FSA3FYYR

#include <iostream>
#include <map>
#include <string>
#include "Halide.h"

using std::cout;
using std::endl;

using namespace Halide;

std::map<std::string, Halide::Internal::Function> get_deps(Func F) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    cout << "Dependencies for " << F.name() << " " << endl;
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        cout << "  .Func " << fit->first << " " << "\n";
        // Func f(fit->second);
        // f.compute_root();
    }
    return flist;
}

void compute_all_root(Func F) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
    }
}

#endif /* end of include guard: GRADIENT_HELPERS_H_FSA3FYYR */

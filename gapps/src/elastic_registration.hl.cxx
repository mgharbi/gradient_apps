#include "gradient_helpers.h"

Var x("x"), y("y"), n("n");

namespace gradient_apps {

class ElasticRegistrationGenerator : public Generator<ElasticRegistrationGenerator> {
public:
    Input<Buffer<float>>  input{"input", 2};
    Input<Buffer<float>>  grid_motion{"grid_motion", 3};
    Output<Buffer<float>> transformed{"transformed", 2};

    void generate() {
        Func f_grid_motion("f_grid_motion");
        f_grid_motion(x, y, n) = grid_motion(x, y, n);
        
        Func interpolated_motion("interpolated_motion");
        Expr fx = x / 32;
        Expr fy = y / 32;
        auto bicubic_function = [](Expr x) -> Expr {
            float a = -0.5f;
            Expr absx = abs(x);
            return select(absx <= 1, (a - 2.f) * absx * absx * absx - (a + 3.f) * absx * absx + 1.f,
                          absx <= 2, a * absx * absx * absx - 5.f * a * absx * absx + 8.f * a * absx - 4.f * a,
                                     0.f);
        };
        Expr interpolated_value = 0.f;
        for (int dy = -1; dy <= 2; dy++) {
            for (int dx = -1; dx <= 2; dx++) {
                interpolated_value = interpolated_value +
                    f_grid_motion(fx + dx, fy + dy, n) *
                    bicubic_function(x - (fx + dx)) *  bicubic_function(y - (fy + dy));
            }
        }
        interpolated_motion(x, y) = interpolated_value;

        // For each point in output, search for interpolated motion and do bilinear look up in input
        Expr ifx = cast<int>(floor(interpolated_motion(x, y, 0)));
        Expr ify = cast<int>(floor(interpolated_motion(x, y, 1)));
        Expr icx = ifx + 1;
        Expr icy = ify + 1;
        Expr wx = interpolated_motion(x, y, 0) - floor(interpolated_motion(x, y, 0));
        transformed(x, y) = input(ifx, ify) * (1.f - wx) * (1.f - wy) +
                            input(icx, ify) *        wx  * (1.f - wy) +
                            input(ifx, icy) * (1.f - wx) *        wy  +
                            input(icx, icy) *        wx  *        wy;

        if(auto_schedule) {
        } else {
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::Conv1dForwardGenerator, conv1d_forward)

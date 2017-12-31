#include <Halide.h>
#include <HalideBuffer.h>
#include <HalideRuntimeCuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include "learnable_demosaick_forward_cuda.h"
// #include "learnable_demosaick_forward.h"

// #include "learnable_demosaick_backward_cuda.h"
// #include "learnable_demosaick_backward.h"

#include "deconv_cg_init_forward_cuda.h"

using Halide::Runtime::Buffer;

int main(int argc, char *argv[])
{
  int w = 256;
  int h = 256;

  int ksize = 11;

  int ret = 0;

  Buffer<float> blurred(w, h, 3);
  Buffer<float> x0(w, h, 3);
  Buffer<float> kernel(ksize, ksize);
  Buffer<float> reg_kernel_weights(5);
  Buffer<float> reg_kernels(5, 5, 5);
  Buffer<float> reg_targets(w, h, 3, 5);
  Buffer<float> precond_kernel(ksize, ksize);
  Buffer<float> w_kernel(w, h, 3);
  Buffer<float> w_reg_kernels(w, h, 3, 5);
  Buffer<float> xrp(w, h, 3, 4);

  for (int n = 0; n < 4; n++) {
      for (int c = 0; c < 3; c++) {
          for (int y = 0; y < h; y++) {
              for (int x = 0; x < w; x++) {
                  xrp(x, y, c, n) = 10.f;
              }
          }
      }
  }
  x0(0, 0, 0) = 50.f;

  printf("running backward\n");
  ret = deconv_cg_init_forward_cuda(
          nullptr, blurred, x0, kernel, reg_kernel_weights, reg_kernels, reg_targets, precond_kernel, w_kernel, w_reg_kernels, xrp);
  printf("done gpu %d\n", ret);
  //for (int i = 0; i < 10; i++) {
      //printf("xrp %f\n", xrp(i, 0, 0, 0));
  //}
 
  return 0;
}

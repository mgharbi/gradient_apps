#include <Halide.h>
#include <HalideBuffer.h>
#include <HalideRuntimeCuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include "learnable_demosaick_forward_cuda.h"
// #include "learnable_demosaick_forward.h"

#include "learnable_demosaick_backward_cuda.h"
// #include "learnable_demosaick_backward.h"

using Halide::Runtime::Buffer;

int main(int argc, char *argv[])
{
  int w = 128;
  int h = 128;
  int n = 64;

  int ksize = 5;
  int k = 8;

  int ret = 0;

  Buffer<float> mosaick(w, h, n);
  Buffer<float> f1(ksize, ksize, k);
  Buffer<float> f2(ksize, ksize, k);
  Buffer<float> output(w, h, 3, n);

  Buffer<float> d_mosaick(w, h, n);
  Buffer<float> d_f1(ksize, ksize, k);
  Buffer<float> d_f2(ksize, ksize, k);
  Buffer<float> d_output(w, h, 3, n);

  // printf("running forward\n");
  // ret = learnable_demosaick_forward_cuda(mosaick, f1, f2, output);
  // printf("done gpu %d\n", ret);
  // ret = learnable_demosaick_forward(mosaick, f1, f2, output);
  // printf("done cpu %d\n", ret);


  printf("running backward\n");
  ret = learnable_demosaick_backward_cuda(
      mosaick, f1, f2, d_output,
      d_mosaick, d_f1, d_f2);
  printf("done gpu %d\n", ret);
  // ret = learnable_demosaick_backward(
  //     mosaick, f1, f2, d_output,
  //     d_mosaick, d_f1, d_f2);
  // printf("done cpu %d\n", ret);
  
  return 0;
}

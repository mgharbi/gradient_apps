#include <Halide.h>
#include <HalideBuffer.h>
#include <HalideRuntimeCuda.h>

#include "conv1d_manual_backward.h"

using Halide::Runtime::Buffer;

int main(int argc, char *argv[])
{
  int w = 128;
  int h = 128;
  int c = 3;

  int ksize = 5;


  Buffer<float> input(w, h, c);
  Buffer<float> filter(ksize, c, c);
  Buffer<float> d_output(w, h, c);
  Buffer<float> d_input(w, h, c);

  printf("running backward\n");
  int ret = conv1d_manual_backward(
      input, filter, d_output, d_input);
  return 0;
}

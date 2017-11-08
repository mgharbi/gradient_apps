#include "Halide.h"

#include "soft_histogram_forward.h"
#include "soft_histogram_backward.h"
#include "histogram_tmp.h"

using Halide::Runtime::Buffer;

int main(int argc, char *argv[])
{
  const int nbins = 8;
  const int size = 64;
  Buffer<float> input(size, size);
  Buffer<float> output(nbins);
  Buffer<float> grad_input(size, size);
  Buffer<float> grad_output(nbins);


  soft_histogram_forward(input, nbins, output);
  soft_histogram_backward(input, grad_output, nbins, grad_input);
  histogram_tmp(input, grad_output, nbins, grad_input);
  return 0;
}

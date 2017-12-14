#include <Halide.h>
#include <HalideBuffer.h>
#include <HalideRuntimeCuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "learnable_demosaick_forward_cuda.h"
// #include "learnable_demosaick_forward.h"

using Halide::Runtime::Buffer;

void wrap(Buffer<float> &buf) {
  int sz = 1;
  for(int i = 0; i < buf.dimensions(); ++i) {
    sz *= buf.extent(i);
  }

  const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();

  float *pData = nullptr;
  cudaError_t res;
  int err;
  res = cudaMalloc((void**)&pData, sizeof(float)*sz);
  err = buf.device_wrap_native(cuda_interface, (uint64_t)pData);
  fprintf(stderr, "wrap %d %d\n", res, err);
  // buf.set_device_dirty();
}

int main(int argc, char *argv[])
{
  int w = 384;
  int h = 640;
  int n = 1;

  int ksize = 5;
  int k = 8;

  Buffer<float> mosaick(w, h, n);
  Buffer<float> f1(ksize, ksize, k);
  Buffer<float> f2(ksize, ksize, k);
  Buffer<float> output(w, h, 3, n);

  wrap(mosaick);
  wrap(f1);
  wrap(f2);
  wrap(output);

  printf("running\n");
  int ret = learnable_demosaick_forward_cuda(mosaick, f1, f2, output);
  printf("done gpu %d\n", ret);
  
  return 0;
}

#include <TH/TH.h>
#include <stdio.h>
#include <memory>

#include "HalideBuffer.h"
#include "dummy.h"

using Halide::Runtime::Buffer;

extern "C" {

int dummy_forward(THFloatTensor *data, THFloatTensor* output) {
  THArgCheck(THFloatTensor_nDimension(data) == 4, 0, "input tensor should be 4D");

  data = THFloatTensor_newContiguous(data); // grab a reference with contiguous memory

  // Wrap in Halide buffers
  int batch_size = THFloatTensor_size(data, 0);
  int channels = THFloatTensor_size(data, 1);
  int height = THFloatTensor_size(data, 2);
  int width = THFloatTensor_size(data, 3);
  THFloatTensor_resize4d(output, batch_size, channels, height, width); 

  // grab a reference with contiguous memory
  float* pInput = THFloatTensor_data(data);
  float* pOutput = THFloatTensor_data(output);

  Buffer<float> in_buf(pInput, {width, height, channels, batch_size});
  Buffer<float> out_buf(pOutput, {width, height, channels, batch_size});

  // Run Halide code
  dummy(in_buf, out_buf);

  THFloatTensor_free(data); // release reference
  return 1;
}

}

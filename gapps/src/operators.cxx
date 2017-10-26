#include <TH/TH.h>
#include <stdio.h>
#include <memory>

#include "HalideBuffer.h"
#include "dummy.h"
#include "bilateral_slice_forward.h"

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
  return 0;
}

int bilateral_slice_forward_(THFloatTensor *grid, THFloatTensor* guide, THFloatTensor* output) {
  THArgCheck(THFloatTensor_nDimension(grid) == 5, 0, "grid should be 5D");
  THArgCheck(THFloatTensor_nDimension(guide) == 3, 0, "grid should be 3D");

  grid = THFloatTensor_newContiguous(grid); // grab a reference with contiguous memory
  guide = THFloatTensor_newContiguous(guide); // grab a reference with contiguous memory

  // Wrap in Halide buffers
  int batch_size = THFloatTensor_size(grid, 0);
  int channels = THFloatTensor_size(grid, 1);
  int gdepth = THFloatTensor_size(grid, 2);
  int gheight = THFloatTensor_size(grid, 3);
  int gwidth = THFloatTensor_size(grid, 4);

  int guide_bs = THFloatTensor_size(guide, 0);
  THArgCheck(guide_bs == batch_size, 0,
             "guide and grid should have same batch_size", guide_bs, batch_size);
  int height = THFloatTensor_size(guide, 1);
  int width = THFloatTensor_size(guide, 2);
  THFloatTensor_resize4d(output, batch_size, channels, height, width); 

  // grab a reference with contiguous memory
  float* pGrid = THFloatTensor_data(grid);
  float* pGuide = THFloatTensor_data(guide);
  float* pOutput = THFloatTensor_data(output);

  Buffer<float> grid_buf(pGrid, {gwidth, gheight, gdepth, channels, batch_size});
  Buffer<float> guide_buf(pGuide, {width, height, channels, batch_size});
  Buffer<float> out_buf(pOutput, {width, height, channels, batch_size});

  // Run Halide code
  bilateral_slice_forward(grid_buf, guide_buf, out_buf);

  THFloatTensor_free(grid); // release reference
  THFloatTensor_free(guide); // release reference
  return 0;
}
}

#include <TH/TH.h>
#include <stdio.h>
#include <memory>
#include <vector>

#include "HalideBuffer.h"
// #include "dummy.h"
// #include "bilateral_slice_forward.h"
#include "bilateral_layer_forward.h"
// #include "bilateral_layer_backward.h"

using Halide::Runtime::Buffer;
using std::vector;

Buffer<float> wrap(THFloatTensor* tensor) {
  int ndims = THFloatTensor_nDimension(tensor);
  vector<int> dims(ndims, 0);
  for(int dim = 0; dim < ndims; ++dim) {
    dims[dim] = THFloatTensor_size(tensor, ndims-1-dim);
  }
  float* pData  = THFloatTensor_data(tensor);
  Buffer<float> buffer(pData, dims);
  return buffer;
}

extern "C" {

// int dummy_forward(THFloatTensor *data, THFloatTensor* output) {
//   THArgCheck(THFloatTensor_nDimension(data) == 4, 0, "input tensor should be 4D");
//
//   data = THFloatTensor_newContiguous(data); // grab a reference with contiguous memory
//
//   // Wrap in Halide buffers
//   int batch_size = THFloatTensor_size(data, 0);
//   int channels = THFloatTensor_size(data, 1);
//   int height = THFloatTensor_size(data, 2);
//   int width = THFloatTensor_size(data, 3);
//   THFloatTensor_resize4d(output, batch_size, channels, height, width); 
//
//   // grab a reference with contiguous memory
//   float* pInput = THFloatTensor_data(data);
//   float* pOutput = THFloatTensor_data(output);
//
//   Buffer<float> in_buf(pInput, {width, height, channels, batch_size});
//   Buffer<float> out_buf(pOutput, {width, height, channels, batch_size});
//
//   // Run Halide code
//   dummy(in_buf, out_buf);
//
//   THFloatTensor_free(data); // release reference
//   return 0;
// }
//
// int bilateral_slice_forward_(THFloatTensor *grid, THFloatTensor* guide, THFloatTensor* output) {
//   THArgCheck(THFloatTensor_nDimension(grid) == 5, 0, "grid should be 5D");
//   THArgCheck(THFloatTensor_nDimension(guide) == 3, 0, "guide should be 3D");
//
//   grid = THFloatTensor_newContiguous(grid); // grab a reference with contiguous memory
//   guide = THFloatTensor_newContiguous(guide); // grab a reference with contiguous memory
//
//   // Wrap in Halide buffers
//   int batch_size = THFloatTensor_size(grid, 0);
//   int channels = THFloatTensor_size(grid, 1);
//   int gdepth = THFloatTensor_size(grid, 2);
//   int gheight = THFloatTensor_size(grid, 3);
//   int gwidth = THFloatTensor_size(grid, 4);
//
//   int guide_bs = THFloatTensor_size(guide, 0);
//   THArgCheck(guide_bs == batch_size, 0,
//              "guide and grid should have same batch_size", guide_bs, batch_size);
//   int height = THFloatTensor_size(guide, 1);
//   int width = THFloatTensor_size(guide, 2);
//   THFloatTensor_resize4d(output, batch_size, channels, height, width); 
//
//   // grab a reference with contiguous memory
//   float* pGrid = THFloatTensor_data(grid);
//   float* pGuide = THFloatTensor_data(guide);
//   float* pOutput = THFloatTensor_data(output);
//
//   Buffer<float> grid_buf(pGrid, {gwidth, gheight, gdepth, channels, batch_size});
//   Buffer<float> guide_buf(pGuide, {width, height, channels, batch_size});
//   Buffer<float> out_buf(pOutput, {width, height, channels, batch_size});
//
//   // Run Halide code
//   bilateral_slice_forward(grid_buf, guide_buf, out_buf);
//
//   THFloatTensor_free(grid); // release reference
//   THFloatTensor_free(guide); // release reference
//   return 0;
// }

/**
 * input [bs, ci, h, w]
 * guide [bs, h, w]
 * kernels [co, ci, gd, gh, gw] 
 * bias [ci, gd]
 */
int bilateral_layer_forward_(THFloatTensor *input,
                            THFloatTensor *guide,
                            THFloatTensor *filter,
                            THFloatTensor *output,
                            const int sigma_x, 
                            const int sigma_y,
                            const int sigma_z) {

  THArgCheck(THFloatTensor_nDimension(input)  == 4, 0, "input should be 4D");
  int batch_size  = THFloatTensor_size(input, 0);
  int channels_in = THFloatTensor_size(input, 1);
  int height      = THFloatTensor_size(input, 2);
  int width       = THFloatTensor_size(input, 3);

  THArgCheck(THFloatTensor_nDimension(guide)  == 3, 0, "grid should be 3D");
  int guide_bs = THFloatTensor_size(guide, 0);
  int guide_h  = THFloatTensor_size(guide, 1);
  int guide_w  = THFloatTensor_size(guide, 2);

  THArgCheck(THFloatTensor_nDimension(filter) == 5, 0, "filter should be 5D");
  int filt_co     = THFloatTensor_size(filter, 0);
  int filt_ci     = THFloatTensor_size(filter, 1);
  // int filt_depth  = THFloatTensor_size(filter, 2);
  int filt_height = THFloatTensor_size(filter, 3);
  int filt_width  = THFloatTensor_size(filter, 4);

  THArgCheck(
      guide_bs == batch_size, 0,
      "guide (%d) and input (%d) should have same batch_size", guide_bs, batch_size);
  THArgCheck(
      guide_h == height, 0,
      "guide (%d) and input (%d) should have same height", guide_h, height);
  THArgCheck(
      guide_w == width, 0,
      "guide (%d) and input (%d) should have same width", guide_w, width);

  THArgCheck(
      filt_ci == channels_in, 0,
      "filter (%d) and input (%d) should have same input channel size", filt_ci, channels_in);  
  
  // grab references with contiguous memory
  input  = THFloatTensor_newContiguous(input);
  guide  = THFloatTensor_newContiguous(guide);
  filter = THFloatTensor_newContiguous(filter);

  THFloatTensor_resize4d(output, batch_size, filt_co, height-filt_height, width-filt_width); 
  // TODO: 0-padding on/off

  // Wrap in Halide buffers
  Buffer<float> input_buf  = wrap(input);
  Buffer<float> guide_buf  = wrap(guide);
  Buffer<float> filter_buf = wrap(filter);
  Buffer<float> output_buf = wrap(output);

  // Run Halide code
  bilateral_layer_forward(
      sigma_x, sigma_y, sigma_z,
      input_buf, guide_buf, filter_buf, output_buf);

  // release reference
  THFloatTensor_free(input);
  THFloatTensor_free(guide);
  THFloatTensor_free(filter);

  return 0;
}

// int bilateral_layer_backward_(THFloatTensor *input,
//                              THFloatTensor *guide,
//                              THFloatTensor *filter,
//                              THFloatTensor *bias,
//                              THFloatTensor *adjoint,
//                              THFloatTensor *d_input,
//                              THFloatTensor *d_guide,
//                              THFloatTensor *d_filter,
//                              THFloatTensor *d_bias) {
//   THArgCheck(THFloatTensor_nDimension(input) == 4, 0, "input should be 4D");
//   THArgCheck(THFloatTensor_nDimension(guide) == 3, 0, "grid should be 3D");
//   THArgCheck(THFloatTensor_nDimension(filter) == 5, 0, "filter should be 5D");
//   THArgCheck(THFloatTensor_nDimension(bias) == 2, 0, "bias should be 2D");
//   THArgCheck(THFloatTensor_nDimension(adjoint) == 4, 0, "adjoint should be 4D");
//
//   input   = THFloatTensor_newContiguous(input); // grab a reference with contiguous memory
//   guide   = THFloatTensor_newContiguous(guide); // grab a reference with contiguous memory
//   filter  = THFloatTensor_newContiguous(filter); // grab a reference with contiguous memory
//   bias    = THFloatTensor_newContiguous(bias); // grab a reference with contiguous memory
//   adjoint = THFloatTensor_newContiguous(adjoint); // grab a reference with contiguous memory
//
//   // Wrap in Halide buffers
//   int batch_size = THFloatTensor_size(input, 0);
//   int channels_in = THFloatTensor_size(input, 1);
//   int height = THFloatTensor_size(input, 2);
//   int width = THFloatTensor_size(input, 3);
//
//   int guide_bs = THFloatTensor_size(guide, 0);
//   THArgCheck(guide_bs == batch_size, 0,
//              "guide and input should have same batch_size", guide_bs, batch_size);
//   int guide_h = THFloatTensor_size(guide, 1);
//   THArgCheck(guide_h == height, 0,
//              "guide and input should have same width and height", guide_h, height);
//   int guide_w = THFloatTensor_size(guide, 2);
//   THArgCheck(guide_w == width, 0,
//              "guide and input should have same width and height", guide_w, width);
//
//   int filter_channels_in = THFloatTensor_size(filter, 0);
//   THArgCheck(filter_channels_in == channels_in, 0,
//              "filter and input should have same input channel size", filter_channels_in, channels_in);  
//   int channels_out = THFloatTensor_size(filter, 1);
//   int z_slice = THFloatTensor_size(filter, 2);
//   int filter_height = THFloatTensor_size(filter, 3);
//   int filter_width = THFloatTensor_size(filter, 4);
//   int bias_bs = THFloatTensor_size(bias, 0);
//   THArgCheck(bias_bs == batch_size, 0,
//              "bias and input should have same batch_size", bias_bs, batch_size);
//   int z_slice_bias = THFloatTensor_size(bias, 1);
//   THArgCheck(z_slice == z_slice_bias, 0,
//              "filter and bias should have same slice size", z_slice, z_slice_bias);  
//
//   int adjoint_bs = THFloatTensor_size(adjoint, 0);
//   THArgCheck(adjoint_bs == batch_size, 0,
//              "adjoint and input should have same batch_size", adjoint_bs, batch_size);
//   int adjoint_channels = THFloatTensor_size(adjoint, 1);
//   THArgCheck(adjoint_channels == channels_out, 0,
//              "adjoint and filter should have same output channel size", adjoint_channels, channels_out);
//   int adjoint_height = THFloatTensor_size(adjoint, 2);
//   THArgCheck(adjoint_height == height - filter_height, 0,
//              "adjoint and output should have same height", adjoint_height, height - filter_height);
//   int adjoint_width = THFloatTensor_size(adjoint, 3);
//   THArgCheck(adjoint_width == width - filter_width, 0,
//              "adjoint and output should have same height", adjoint_width, width - filter_width);
//
//   THFloatTensor_resize4d(d_input, batch_size, channels_in, height, width);
//   THFloatTensor_resize3d(d_guide, batch_size, height, width);
//   THFloatTensor_resize5d(d_filter, batch_size, channels_out, channels_in, filter_height, filter_width);
//   THFloatTensor_resize2d(d_bias, batch_size, z_slice);
//
//   // grab a reference with contiguous memory
//   float* pInput = THFloatTensor_data(input);
//   float* pGuide = THFloatTensor_data(guide);
//   float* pFilter = THFloatTensor_data(filter);
//   float* pBias = THFloatTensor_data(bias);
//   float* pAdjoint = THFloatTensor_data(adjoint);
//   // float* pDInput = THFloatTensor_data(d_input);
//   // float* pDGuide = THFloatTensor_data(d_guide);
//   // float* pDFilter = THFloatTensor_data(d_filter);
//   // float* pDBias = THFloatTensor_data(d_bias);
//
//   Buffer<float> input_buf(pInput, {width, height, channels_in, batch_size});
//   Buffer<float> guide_buf(pGuide, {width, height, batch_size});
//   Buffer<float> filter_buf(pFilter, {filter_width, filter_height, z_slice, channels_in, channels_out});
//   Buffer<float> bias_buf(pBias, {z_slice, batch_size});
//   Buffer<float> adjoint_buf(pAdjoint, {width - filter_width,
//                                        height - filter_height,
//                                        channels_out,
//                                        batch_size});
//   Buffer<float> d_input_buf(pInput, {width, height, channels_in, batch_size});
//   Buffer<float> d_guide_buf(pGuide, {width, height, batch_size});
//   Buffer<float> d_filter_buf(pFilter, {filter_width, filter_height, z_slice, channels_in, channels_out});
//   Buffer<float> d_bias_buf(pBias, {z_slice, batch_size});
//
//   // Run Halide code
//   bilateral_layer_backward(input_buf, guide_buf, filter_buf, bias_buf, adjoint_buf,
//                            d_input_buf, d_guide_buf, d_filter_buf, d_bias_buf);
//
//   THFloatTensor_free(input); // release reference
//   THFloatTensor_free(guide); // release reference
//   THFloatTensor_free(filter); // release reference
//   THFloatTensor_free(bias); // release reference
//   THFloatTensor_free(adjoint); // release reference
//   // Should we release reference of output ??
//   return 0;
// }

}

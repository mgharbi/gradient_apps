#include <TH/TH.h>
#include <stdio.h>
#include <memory>
#include <vector>

#include "HalideBuffer.h"
#include "playground_forward.h"
#include "playground_backward.h"
#include "bilateral_layer_forward.h"
#include "bilateral_layer_backward.h"
#include "ahd_demosaick_forward.h"

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

int playground_forward_(THFloatTensor *data1, THFloatTensor *data2, THFloatTensor* output) {
  THArgCheck(THFloatTensor_nDimension(data1) == 4, 0, "input tensor should be 4D");
  int batch_size = THFloatTensor_size(data1, 0);
  int channels = THFloatTensor_size(data1, 1);
  int height = THFloatTensor_size(data1, 2);
  int width = THFloatTensor_size(data1, 3);
  THArgCheck(THFloatTensor_nDimension(data2) == 4, 0, "input tensor should be 4D");

  // grab a reference with contiguous memory
  data1 = THFloatTensor_newContiguous(data1);
  data2 = THFloatTensor_newContiguous(data2);

  THFloatTensor_resize4d(output, batch_size, channels, height, width); 

  // Wrap in Halide buffers
  Buffer<float> data1_buf  = wrap(data1);
  Buffer<float> data2_buf  = wrap(data2);
  Buffer<float> output_buf = wrap(output);

  // Run Halide code
  playground_forward(data1_buf, data2_buf, output_buf);

  THFloatTensor_free(data1); // release reference
  THFloatTensor_free(data2); // release reference
  return 0;
}

int playground_backward_(
    THFloatTensor *data1, THFloatTensor *data2, THFloatTensor* d_output,
    THFloatTensor *d_data1, THFloatTensor *d_data2) {
  THArgCheck(THFloatTensor_nDimension(data1) == 4, 0, "input tensor should be 4D");
  int batch_size = THFloatTensor_size(data1, 0);
  int channels = THFloatTensor_size(data1, 1);
  int height = THFloatTensor_size(data1, 2);
  int width = THFloatTensor_size(data1, 3);
  THArgCheck(THFloatTensor_nDimension(data2) == 4, 0, "input tensor should be 4D");
  THArgCheck(THFloatTensor_nDimension(d_output) == 4, 0, "output tensor should be 4D");

  // grab a reference with contiguous memory
  data1 = THFloatTensor_newContiguous(data1);
  data2 = THFloatTensor_newContiguous(data2);
  d_output = THFloatTensor_newContiguous(d_output);

  THFloatTensor_resize4d(d_data1, batch_size, channels, height, width); 
  THFloatTensor_resize4d(d_data2, batch_size, channels, height, width); 

  // Wrap in Halide buffers
  Buffer<float> data1_buf  = wrap(data1);
  Buffer<float> data2_buf  = wrap(data2);
  Buffer<float> d_output_buf = wrap(d_output);
  Buffer<float> d_data1_buf  = wrap(d_data1);
  Buffer<float> d_data2_buf  = wrap(d_data2);

  // Run Halide code
  playground_backward(data1_buf, data2_buf, d_output_buf, d_data1_buf, d_data2_buf);

  THFloatTensor_free(data1); // release reference
  THFloatTensor_free(data2); // release reference
  THFloatTensor_free(d_output); // release reference
  return 0;
}


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

int bilateral_layer_backward_(THFloatTensor *input,
                             THFloatTensor *guide,
                             THFloatTensor *filter,
                             THFloatTensor *d_output,
                             THFloatTensor *d_input,
                             THFloatTensor *d_guide,
                             THFloatTensor *d_filter,
                             const int sigma_x,
                             const int sigma_y,
                             const int sigma_z) {
  THArgCheck(THFloatTensor_nDimension(input) == 4, 0, "input should be 4D");
  int batch_size  = THFloatTensor_size(input, 0);
  int channels_in = THFloatTensor_size(input, 1);
  int height      = THFloatTensor_size(input, 2);
  int width       = THFloatTensor_size(input, 3);

  THArgCheck(THFloatTensor_nDimension(guide) == 3, 0, "grid should be 3D");
  int guide_bs = THFloatTensor_size(guide, 0);
  int guide_h  = THFloatTensor_size(guide, 1);
  int guide_w  = THFloatTensor_size(guide, 2);

  THArgCheck(THFloatTensor_nDimension(filter) == 5, 0, "filter should be 5D");
  int filt_co     = THFloatTensor_size(filter, 0);
  int filt_ci     = THFloatTensor_size(filter, 1);
  int filt_depth  = THFloatTensor_size(filter, 2);
  int filt_height = THFloatTensor_size(filter, 3);
  int filt_width  = THFloatTensor_size(filter, 4);

  THArgCheck(THFloatTensor_nDimension(d_output) == 4, 0, "d_output should be 4D");
  int adjoint_bs       = THFloatTensor_size(d_output, 0);
  int adjoint_channels = THFloatTensor_size(d_output, 1);
  int adjoint_height   = THFloatTensor_size(d_output, 2);
  int adjoint_width    = THFloatTensor_size(d_output, 3);

  THArgCheck(
      guide_bs == batch_size, 0,
      "guide (%d) and input (%d) should have same batch_size",
      guide_bs, batch_size);
  THArgCheck(
      guide_h == height, 0,
      "guide (%d) and input (%d) should have same height", guide_h, height);
  THArgCheck(
      guide_w == width, 0,
      "guide (%d) and input (%d) should have same width", guide_w, width);

  THArgCheck(
      filt_ci == channels_in, 0,
      "filter (%d) and input (%d) should have same input channel size",
      filt_ci, channels_in);  

  THArgCheck(
      adjoint_bs == batch_size, 0,
      "adjoint (%d) and input (%d) should have same batch_size",
      adjoint_bs, batch_size);
  THArgCheck(
      adjoint_channels == filt_co, 0,
      "adjoint (%d) and filter (%d) should have same output channel size",
      adjoint_channels, filt_co);
  THArgCheck(
      adjoint_height == height - filt_height, 0,
      "adjoint (%d) and output (%d) should have same height", 
      adjoint_height, height - filt_height);
  THArgCheck(
      adjoint_width == width - filt_width, 0,
      "adjoint (%d) and output (%d) should have same height",
      adjoint_width, width - filt_width);

  // grab references with contiguous memory
  input   = THFloatTensor_newContiguous(input);
  guide   = THFloatTensor_newContiguous(guide);
  filter  = THFloatTensor_newContiguous(filter);
  d_output = THFloatTensor_newContiguous(d_output);

  THFloatTensor_resize4d(d_input, batch_size, channels_in, height, width);
  THFloatTensor_resize3d(d_guide, batch_size, height, width);
  THFloatTensor_resize5d(d_filter, filt_co, filt_ci, filt_depth, filt_height, filt_width);

  // Wrap in Halide buffers
  Buffer<float> input_buf  = wrap(input);
  Buffer<float> guide_buf  = wrap(guide);
  Buffer<float> filter_buf = wrap(filter);
  Buffer<float> d_output_buf = wrap(d_output);
  Buffer<float> d_input_buf = wrap(d_input);
  Buffer<float> d_guide_buf = wrap(d_guide);
  Buffer<float> d_filter_buf = wrap(d_filter);

  // Run Halide code
  bilateral_layer_backward(
      sigma_x, sigma_y, sigma_z,
      input_buf, guide_buf, filter_buf, d_output_buf,
      d_input_buf, d_guide_buf, d_filter_buf) ;

  // release reference
  THFloatTensor_free(input);
  THFloatTensor_free(guide);
  THFloatTensor_free(filter);
  THFloatTensor_free(d_output);
  return 0;
}

int ahd_demosaick_forward_(THFloatTensor *mosaick, THFloatTensor* output) {
  THArgCheck(THFloatTensor_nDimension(mosaick) == 2, 0, "mosaick tensor should be 4D");
  int height = THFloatTensor_size(mosaick, 0);
  int width = THFloatTensor_size(mosaick, 1);

  // grab a reference with contiguous memory
  mosaick = THFloatTensor_newContiguous(mosaick);

  THFloatTensor_resize3d(output, 3, height, width); 

  // Wrap in Halide buffers
  Buffer<float> mosaick_buf  = wrap(mosaick);
  Buffer<float> output_buf = wrap(output);

  // Run Halide code
  ahd_demosaick_forward(mosaick_buf, output_buf);

  THFloatTensor_free(mosaick); // release reference
  return 0;
}

} // extern C


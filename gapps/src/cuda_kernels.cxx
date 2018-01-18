#include <TH/TH.h>
#include <THC/THC.h>
#include <stdio.h>

#include "cuda_kernels/bilateral_slice.h"

extern THCState *state;

extern "C" {

int bilateral_slice_manual_forward(
    THCudaTensor *_grid, THCudaTensor *_guide, 
    THCudaTensor *_input, THCudaTensor *_out) {

  _grid = THCudaTensor_newContiguous(state, _grid);
  _guide = THCudaTensor_newContiguous(state, _guide);
  _input = THCudaTensor_newContiguous(state, _input);
  _out = THCudaTensor_newContiguous(state, _out);

  int bs = THCudaTensor_size(state, _grid, 0);
  int chans = THCudaTensor_size(state, _grid, 1);
  int gd = THCudaTensor_size(state, _grid, 2);
  int gh = THCudaTensor_size(state, _grid, 3);
  int gw = THCudaTensor_size(state, _grid, 4);

  int ci = THCudaTensor_size(state, _input, 1);
  int h = THCudaTensor_size(state, _input, 2);
  int w = THCudaTensor_size(state, _input, 3);

  int co = chans / (ci + 1);

  THArgCheck(
      chans % (ci + 1) == 0
      , 0, "Bilateral grid and input channels incompatible");
  THArgCheck(
      bs == THCudaTensor_size(state, _guide, 0) &&
      h == THCudaTensor_size(state, _guide, 1) &&
      w == THCudaTensor_size(state, _guide, 2)
      , 1, "Guide dimensions incorrect");
  THArgCheck(
      bs == THCudaTensor_size(state, _input, 0)
      , 2, "Input dimensions incorrect");

  THCudaTensor_resize4d(state, _out, bs, co, h, w);

  float *pGrid = THCudaTensor_data(state, _grid);
  float *pGuide = THCudaTensor_data(state, _guide);
  float *pInput = THCudaTensor_data(state, _input);
  float *pOutput = THCudaTensor_data(state, _out);

  BilateralSliceApplyKernelLauncher(
      bs, gh, gw, gd, ci, co, h, w,
      pGrid, pGuide, pInput, pOutput);

  THCudaTensor_free(state, _grid);
  THCudaTensor_free(state, _guide);
  THCudaTensor_free(state, _input);
  THCudaTensor_free(state, _out);

  return 0;
}

int bilateral_slice_manual_backward(
    THCudaTensor *_grid, THCudaTensor *_guide, 
    THCudaTensor *_input, THCudaTensor *_d_out,
    THCudaTensor *_d_grid, THCudaTensor *_d_guide, 
    THCudaTensor *_d_input) {
  _grid = THCudaTensor_newContiguous(state, _grid);
  _guide = THCudaTensor_newContiguous(state, _guide);
  _input = THCudaTensor_newContiguous(state, _input);
  _d_out = THCudaTensor_newContiguous(state, _d_out);

  _d_grid = THCudaTensor_newContiguous(state, _d_grid);
  _d_guide = THCudaTensor_newContiguous(state, _d_guide);
  _d_input = THCudaTensor_newContiguous(state, _d_input);

  int bs = THCudaTensor_size(state, _grid, 0);
  int chans = THCudaTensor_size(state, _grid, 1);
  int gd = THCudaTensor_size(state, _grid, 2);
  int gh = THCudaTensor_size(state, _grid, 3);
  int gw = THCudaTensor_size(state, _grid, 4);

  int ci = THCudaTensor_size(state, _input, 1);
  int h = THCudaTensor_size(state, _input, 2);
  int w = THCudaTensor_size(state, _input, 3);

  int co = chans / (ci + 1);

  THArgCheck(
      chans % (ci + 1) == 0
      , 0, "Bilateral grid and input channels incompatible");
  THArgCheck(
      bs == THCudaTensor_size(state, _guide, 0) &&
      h == THCudaTensor_size(state, _guide, 1) &&
      w == THCudaTensor_size(state, _guide, 2)
      , 1, "Guide dimensions incorrect");
  THArgCheck(
      bs == THCudaTensor_size(state, _input, 0)
      , 2, "Input dimensions incorrect");

  THCudaTensor_resize5d(state, _d_grid, bs, chans, gd, gh, gw);
  THCudaTensor_resize3d(state, _d_guide, bs, h, w);
  THCudaTensor_resize4d(state, _d_input, bs, ci, h, w);

  float *pGrid = THCudaTensor_data(state, _grid);
  float *pGuide = THCudaTensor_data(state, _guide);
  float *pInput = THCudaTensor_data(state, _input);
  float *pDOutput = THCudaTensor_data(state, _d_out);

  float *pDGrid = THCudaTensor_data(state, _d_grid);
  float *pDGuide = THCudaTensor_data(state, _d_guide);
  float *pDInput = THCudaTensor_data(state, _d_input);

  BilateralSliceApplyGradKernelLauncher(
      bs, gh, gw, gd, ci, co, h, w,
      pGrid, pGuide, pInput, pDOutput,
      pDGrid, pDGuide, pDInput);

  THCudaTensor_free(state, _grid);
  THCudaTensor_free(state, _guide);
  THCudaTensor_free(state, _input);
  THCudaTensor_free(state, _d_out);

  THCudaTensor_free(state, _d_grid);
  THCudaTensor_free(state, _d_guide);
  THCudaTensor_free(state, _d_input);

  return 0;
}

}  // extern "C"

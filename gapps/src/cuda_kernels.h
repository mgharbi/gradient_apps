int bilateral_slice_manual_forward(
    THCudaTensor *_grid, THCudaTensor *_guide, 
    THCudaTensor *_input, THCudaTensor *_out);

int bilateral_slice_manual_backward(
    THCudaTensor *_grid, THCudaTensor *_guide, 
    THCudaTensor *_input, THCudaTensor *_d_out,
    THCudaTensor *_d_grid, THCudaTensor *_d_guide, 
    THCudaTensor *_d_input);

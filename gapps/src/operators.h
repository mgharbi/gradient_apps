int dummy_forward(THFloatTensor *data, THFloatTensor *output);

int bilateral_slice_forward_(THFloatTensor *grid, THFloatTensor *guide, THFloatTensor *output);

int bilateral_layer_forward(THFloatTensor *input,
                            THFloatTensor *guide,
                            THFloatTensor *filter,
                            THFloatTensor *bias,
                            THFloatTensor *output);

int bilateral_layer_backward(THFloatTensor *input,
                             THFloatTensor *guide,
                             THFloatTensor *filter,
                             THFloatTensor *bias);
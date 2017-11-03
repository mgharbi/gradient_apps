int playground_forward_(THFloatTensor *data1, THFloatTensor *data2, THFloatTensor *output);
int playground_backward_(
    THFloatTensor *data1, THFloatTensor *data2, THFloatTensor *d_output,
    THFloatTensor *d_data1, THFloatTensor *d_data2);

int bilateral_layer_forward_(THFloatTensor *input,
                            THFloatTensor *guide,
                            THFloatTensor *filter,
                            THFloatTensor *output,
                            const int sigma_x,
                            const int sigma_y,
                            const int sigma_z);

int bilateral_layer_backward_(THFloatTensor *input,
                             THFloatTensor *guide,
                             THFloatTensor *filter,
                             THFloatTensor *d_output,
                             THFloatTensor *d_input,
                             THFloatTensor *d_guide,
                             THFloatTensor *d_filter,
                            const int sigma_x,
                            const int sigma_y,
                            const int sigma_z);

int ahd_demosaick_forward_(THFloatTensor *mosaick, THFloatTensor *output);
// int playground_backward_(
//     THFloatTensor *data1, THFloatTensor *data2, THFloatTensor *d_output,
//     THFloatTensor *d_data1, THFloatTensor *d_data2);

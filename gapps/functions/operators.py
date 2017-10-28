import torch
from torch.autograd import Function
from torch.autograd import Variable
from .._ext import operators as ops


class Dummy(Function):
  """"""

  @staticmethod
  def forward(ctx, data):
    ctx.save_for_backward(data)

    output = data.new()
    ops.dummy_forward(data, output)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    data = ctx.saved_variables

    # not gradient for this op
    grad_data = None

  #   grad_data = data.data.new()
  #
  #   backend(data.data).sample_weighting_backward(
  #       data.data, coords.data, params.data, kernels.data,
  #       grad_output.data, grad_output_w.data,
  #       grad_data, grad_coords, grad_params, grad_kernels, nsize)
  #
  #   grad_data = Variable(grad_data)

    return grad_data


class BilateralSlice(Function):
  """"""

  @staticmethod
  def forward(ctx, grid, guide):
    # ctx.save_for_backward(g)

    output = grid.new()
    ops.bilateral_slice_forward_(grid, guide, output)

    return output

class BilateralLayer(Function):
  """"""

  @staticmethod
  def forward(ctx, input, guide, filter, bias):
    ctx.save_for_backward(input, guide, filter, bias)

    output = input.new()
    ops.bilateral_layer_forward(input, guide, filter, bias, output)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    input, guide, filter, bias = ctx.saved_variables

    d_input = input.new()
    d_guide = guide.new()
    d_filter = filter.new()
    d_bias = bias.new()
    ops.bilateral_layer_backward(input, guide, filter, bias, grad_output,
                                 d_input, d_guide, d_filter, d_bias)

    return d_input, d_guide, d_filter, d_bias

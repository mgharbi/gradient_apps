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

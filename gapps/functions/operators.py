import torch
from torch.autograd import Function
from torch.autograd import Variable
from .._ext import operators as ops


class BilateralLayer(Function):
  """"""

  @staticmethod
  def forward(ctx, input, guide, filter, sigma_x, sigma_y, sigma_z):
    ctx.save_for_backward(input, guide, filter)
    ctx.sigma_x = sigma_x
    ctx.sigma_y = sigma_y
    ctx.sigma_z = sigma_z

    output = input.new()
    ops.bilateral_layer_forward_(
        input, guide, filter, output,
        sigma_x, sigma_y, sigma_z)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input, guide, filter = ctx.saved_variables

    sigma_x = ctx.sigma_x
    sigma_y = ctx.sigma_y
    sigma_z = ctx.sigma_z

    d_input = input.data.new()
    d_guide = guide.data.new()
    d_filter = filter.data.new()
    ops.bilateral_layer_backward_(
        input.data, guide.data, filter.data, d_output.data,
        d_input, d_guide, d_filter,
        sigma_x, sigma_y, sigma_z)

    d_input = Variable(d_input)
    d_guide = Variable(d_guide)
    d_filter = Variable(d_filter)

    return d_input, d_guide, d_filter, None, None, None


class Playground(Function):
  """"""

  @staticmethod
  def forward(ctx, input1, input2):
    ctx.save_for_backward(input1, input2)

    output = input1.new()
    ops.playground_forward_(
        input1, input2, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input1, input2 = ctx.saved_variables

    d_input1 = input1.data.new()
    d_input2 = input2.data.new()
    ops.playground_backward_(
        input1.data, input2.data, d_output.data,
        d_input1, d_input2)

    d_input1 = Variable(d_input1)
    d_input2 = Variable(d_input2)

    return d_input1, d_input2

class AHDDemosaick(Function):
  """"""

  @staticmethod
  def forward(ctx, mosaick):
    ctx.save_for_backward(mosaick)

    output = mosaick.new()
    ops.ahd_demosaick_forward_(
        mosaick, output)

    return output

class Histogram(Function):
  """"""

  @staticmethod
  def forward(ctx, input, nbins):
    ctx.save_for_backward(input)
    ctx.nbins = nbins

    output = input.new()
    ops.histogram_forward_(input, output, nbins)

    return output

  # @staticmethod
  # def backward(ctx, output_grad):
  #   input = ctx.saved_variables[0]
  #   nbins = ctx.nbins
  #
  #   input_grad = output_grad.data.new()
  #   ops.histogram_backward_(input.data, output_grad.data, nbins, input_grad)
  #
  #   input_grad = Variable(input_grad)
  #
  #   return input_grad, None


class SoftHistogram(Function):
  """"""

  @staticmethod
  def forward(ctx, input, nbins):
    ctx.save_for_backward(input)
    ctx.nbins = nbins

    output = input.new()
    ops.soft_histogram_forward_(input, output, nbins)

    return output

  @staticmethod
  def backward(ctx, output_grad):
    input = ctx.saved_variables[0]
    nbins = ctx.nbins

    input_grad = output_grad.data.new()
    ops.soft_histogram_backward_(input.data, output_grad.data, nbins, input_grad)

    input_grad = Variable(input_grad)

    return input_grad, None

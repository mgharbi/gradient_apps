import inspect
import re

import torch
from torch.autograd import Function
from torch.autograd import Variable
from .._ext import operators as ops


def has_cuda_inputs(args):
  for a in args:
    md = inspect.getmodule(a.__class__).__name__
    if "cuda" in md:
      return True
  return False


def wrap_op(op, cuda_op):
  def _func(*args, **kwargs):
    if has_cuda_inputs(args):
      return cuda_op(*args, **kwargs)
    else:
      return op(*args, **kwargs)
  return _func


th_re = re.compile(r"((?!cuda).)*_th_$")
ops_funcs = [f for f in inspect.getmembers(ops, inspect.isfunction) if th_re.match(f[0])]
for op_name, op in ops_funcs:
  wrapper_name = op_name[:-4]  # remove th suffix
  cuda_name = wrapper_name + "_cuda_th_"
  cuda_op = getattr(ops, cuda_name)
  setattr(ops, wrapper_name, wrap_op(op, cuda_op))


class BilateralLayer(Function):
  """"""

  @staticmethod
  def forward(ctx, input, guide, filter, sigma_x, sigma_y, sigma_z):
    ctx.save_for_backward(input, guide, filter)
    ctx.sigma_x = sigma_x
    ctx.sigma_y = sigma_y
    ctx.sigma_z = sigma_z

    bs, ci, h, w = input.shape
    co = filter.shape[0]

    assert guide.shape[0] == bs
    assert guide.shape[1] == h
    assert guide.shape[2] == w
    assert filter.shape[1] == ci

    output = input.new()
    output.resize_(bs, co, h, w);

    ops.bilateral_layer_forward(
        sigma_x, sigma_y, sigma_z,
        input, guide, filter, output)

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
    d_input.resize_as_(input.data)
    d_guide.resize_as_(guide.data)
    d_filter.resize_as_(filter.data)

    ops.bilateral_layer_backward(
        sigma_x, sigma_y, sigma_z,
        input.data, guide.data, filter.data, d_output.data,
        d_input, d_guide, d_filter)

    d_input = Variable(d_input)
    d_guide = Variable(d_guide)
    d_filter = Variable(d_filter)

    return d_input, d_guide, d_filter, None, None, None


# class AHDDemosaick(Function):
#   """"""
#
#   @staticmethod
#   def forward(ctx, mosaick):
#     ctx.save_for_backward(mosaick)
#
#     output = mosaick.new()
#     ops.ahd_demosaick_forward_(
#         mosaick, output)
#
#     return output
#
# class Histogram(Function):
#   """"""
#
#   @staticmethod
#   def forward(ctx, input, nbins):
#     ctx.save_for_backward(input)
#     ctx.nbins = nbins
#
#     output = input.new()
#     ops.histogram_forward_(input, output, nbins)
#
#     return output
#
#   # @staticmethod
#   # def backward(ctx, output_grad):
#   #   input = ctx.saved_variables[0]
#   #   nbins = ctx.nbins
#   #
#   #   input_grad = output_grad.data.new()
#   #   ops.histogram_backward_(input.data, output_grad.data, nbins, input_grad)
#   #
#   #   input_grad = Variable(input_grad)
#   #
#   #   return input_grad, None
#
#
# class SoftHistogram(Function):
#   """"""
#
#   @staticmethod
#   def forward(ctx, input, nbins):
#     ctx.save_for_backward(input)
#     ctx.nbins = nbins
#
#     output = input.new()
#     ops.soft_histogram_forward_(input, output, nbins)
#
#     return output
#
#   @staticmethod
#   def backward(ctx, output_grad):
#     input = ctx.saved_variables[0]
#     nbins = ctx.nbins
#
#     input_grad = output_grad.data.new()
#     ops.soft_histogram_backward_(input.data, output_grad.data, nbins, input_grad)
#
#     input_grad = Variable(input_grad)
#
#     return input_grad, None

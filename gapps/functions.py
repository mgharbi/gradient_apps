import inspect
import re

import torch
from torch.autograd import Function
from torch.autograd import Variable
from ._ext import operators as ops


def has_cuda_inputs(args):
  for a in args:
    md = inspect.getmodule(a.__class__).__name__
    if "cuda" in md:
      return True
  return False


def wrap_op(op, cuda_op):
  def _func(*args, **kwargs):
    if has_cuda_inputs(args) and cuda_op is not None:
      return cuda_op(*args, **kwargs)
    else:
      return op(*args, **kwargs)
  return _func


th_re = re.compile(r"((?!cuda).)*_th_$")
ops_funcs = [f for f in inspect.getmembers(ops, inspect.isfunction) if th_re.match(f[0])]
for op_name, op in ops_funcs:
  wrapper_name = op_name[:-4]  # remove th suffix
  cuda_name = wrapper_name + "_cuda_th_"
  try:
    cuda_op = getattr(ops, cuda_name)
  except AttributeError:
    print("op {}, not present, setting to none".format(cuda_name))
    cuda_op = None
  setattr(ops, wrapper_name, wrap_op(op, cuda_op))


class Histogram(Function):
  """"""

  @staticmethod
  def forward(ctx, input, nbins):
    ctx.save_for_backward(input)
    ctx.nbins = nbins

    assert nbins > 0
    output = input.new()
    output.resize_(nbins);

    ops.histogram_forward(input, nbins, output)

    return output

  @staticmethod
  def backward(ctx, output_grad):
    input = ctx.saved_variables[0]
    nbins = ctx.nbins

    input_grad = input.data.new()
    input_grad.resize_as_(input.data)
    ops.histogram_backward(input.data, output_grad.data, nbins, input_grad)

    input_grad = Variable(input_grad)

    return input_grad, None


class SoftHistogram(Function):
  """"""

  @staticmethod
  def forward(ctx, input, nbins):
    ctx.save_for_backward(input)
    ctx.nbins = nbins

    assert nbins > 0

    output = input.new()
    output.resize_(nbins);
    ops.soft_histogram_forward(input, nbins, output)

    return output

  @staticmethod
  def backward(ctx, output_grad):
    input = ctx.saved_variables[0]
    nbins = ctx.nbins

    input_grad = input.data.new()
    input_grad.resize_as_(input.data)
    ops.soft_histogram_backward(input.data, output_grad.data, nbins, input_grad)

    input_grad = Variable(input_grad)

    return input_grad, None


class Conv1d(Function):
  """"""

  @staticmethod
  def forward(ctx, input, filter, manual_backward=False):
    ctx.save_for_backward(input, filter)
    ctx.manual_backward = manual_backward

    bs, ci, w = input.shape
    co = filter.shape[0]

    assert filter.shape[1] == ci

    output = input.new()
    output.resize_(bs, co, w);

    ops.conv1d_forward(
        input, filter, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input, filter = ctx.saved_variables

    d_input = input.data.new()
    w, ci, n = input.shape
    d_filter = filter.data.new()
    d_input.resize_(w, ci, n)
    # d_input.resize_as_(input.data)
    d_filter.resize_as_(filter.data)

    if ctx.manual_backward:
      ops.conv1d_manual_backward(
          input.data, filter.data, d_output.data, d_input)
    else:
      ops.conv1d_backward(
          input.data, filter.data, d_output.data,
          d_input)

    d_input = Variable(d_input)
    d_filter = Variable(d_filter)

    return d_input, d_filter, None


class Conv3d(Function):
  """"""

  @staticmethod
  def forward(ctx, input, filter):
    ctx.save_for_backward(input, filter)

    bs, ci, d, h, w = input.shape
    co = filter.shape[0]

    assert filter.shape[1] == ci

    output = input.new()
    output.resize_(bs, co, d, h, w);

    ops.conv3d_forward(
        input, filter, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input, filter = ctx.saved_variables

    d_input = input.data.new()
    d_filter = filter.data.new()
    d_input.resize_as_(input.data)
    d_filter.resize_as_(filter.data)

    ops.conv3d_backward(
        input.data, filter.data, d_output.data,
        d_input, d_filter)

    d_input = Variable(d_input)
    d_filter = Variable(d_filter)

    return d_input, d_filter


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


class NaiveDemosaick(Function):
  """"""

  @staticmethod
  def forward(ctx, mosaick):
    ctx.save_for_backward(mosaick)

    output = mosaick.new()
    bs, ci, h, w = mosaick.shape
    output.resize_(bs, 3, h, w)
    ops.naive_demosaick_forward(
        mosaick.view(bs, h, w), output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    mosaick = ctx.saved_variables[0]

    bs, ci, h, w = mosaick.shape

    d_mosaick = mosaick.data.new()
    d_mosaick.resize_as_(mosaick.data)

    ops.naive_demosaick_backward(
        mosaick.data.view(bs, h, w), d_output.data,
        d_mosaick.view(bs, h, w))

    d_mosaick = Variable(d_mosaick)

    return d_mosaick


class LearnableDemosaick(Function):
  """"""

  @staticmethod
  def forward(ctx, mosaick, selection_filters, green_filters):
    ctx.save_for_backward(mosaick, selection_filters, green_filters)

    output = mosaick.new()
    bs, ci, h, w = mosaick.shape
    assert ci == 1

    output.resize_(bs, 3, h, w)
    ops.learnable_demosaick_forward(
        mosaick.view(bs, h, w), selection_filters, green_filters, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    mosaick, selection_filters, green_filters = ctx.saved_variables

    d_mosaick = mosaick.data.new()
    d_mosaick.resize_as_(mosaick.data)
    d_sel_filts = selection_filters.data.new()
    d_sel_filts.resize_as_(selection_filters.data).zero_()
    d_green_filts = green_filters.data.new()
    d_green_filts.resize_as_(green_filters.data).zero_()

    bs, ci, h, w = mosaick.shape

    ops.learnable_demosaick_backward(
        mosaick.data.view(bs, h, w), selection_filters.data, green_filters.data,
        d_output.data,
        d_mosaick.view(bs, h, w), d_sel_filts, d_green_filts)

    d_mosaick = Variable(d_mosaick)
    d_sel_filts = Variable(d_sel_filts)
    d_green_filts = Variable(d_green_filts)

    return d_mosaick, d_sel_filts, d_green_filts

class DeconvCGInit(Function):
  """"""

  @staticmethod
  def forward(ctx, blurred, x0, kernel, reg_kernel_weights, reg_kernels, reg_target_kernels):
    ctx.save_for_backward(blurred, x0, kernel, reg_kernel_weights, reg_kernels, reg_target_kernels)

    xrp = blurred.new()
    b, ci, h, w = blurred.shape
    assert b == 1

    xrp.resize_(3, ci, h, w)
    ops.deconv_cg_init_forward(
        blurred.view(ci, h, w), x0.view(ci, h, w), kernel, reg_kernel_weights, reg_kernels, reg_target_kernels, xrp)

    return xrp

  @staticmethod
  def backward(ctx, d_xrp):
    blurred, x0, kernel, reg_kernel_weights, reg_kernels, reg_target_kernels = ctx.saved_variables

    d_reg_kernel_weights = reg_kernel_weights.data.new()
    d_reg_kernel_weights.resize_as_(reg_kernel_weights.data)
    d_reg_kernels = reg_kernels.data.new()
    d_reg_kernels.resize_as_(reg_kernels.data)
    d_reg_target_kernels = reg_target_kernels.data.new()
    d_reg_target_kernels.resize_as_(reg_target_kernels.data)

    b, ci, h, w = blurred.shape
    assert b == 1

    ops.deconv_cg_init_backward(
        blurred.data.view(ci, h, w), x0.data, kernel.data,
        reg_kernel_weights.data, reg_kernels.data, reg_target_kernels.data, d_xrp.data,
        d_reg_kernel_weights, d_reg_kernels, d_reg_target_kernels)

    d_reg_kernel_weights = Variable(d_reg_kernel_weights)
    d_reg_kernels = Variable(d_reg_kernels)
    d_reg_target_kernels = Variable(d_reg_target_kernels)

    return None, None, None, d_reg_kernel_weights, d_reg_kernels, d_reg_target_kernels

class DeconvCGIter(Function):
  """"""

  @staticmethod
  def forward(ctx, xrp, kernel, reg_kernel_weights, reg_kernels):
    ctx.save_for_backward(xrp, kernel, reg_kernel_weights, reg_kernels)

    next_xrp = xrp.new()
    n, ci, h, w = xrp.shape
    assert n == 3
    assert ci == 3

    next_xrp.resize_(n, ci, h, w)
    ops.deconv_cg_iter_forward(
        xrp, kernel, reg_kernel_weights, reg_kernels, next_xrp)

    return next_xrp

  @staticmethod
  def backward(ctx, d_next_xrp):
    xrp, kernel, reg_kernel_weights, reg_kernels = ctx.saved_variables

    d_xrp = xrp.data.new()
    d_xrp.resize_as_(xrp.data)
    d_reg_kernel_weights = reg_kernel_weights.data.new()
    d_reg_kernel_weights.resize_as_(reg_kernel_weights.data)
    d_reg_kernels = reg_kernels.data.new()
    d_reg_kernels.resize_as_(reg_kernels.data)

    ops.deconv_cg_iter_backward(
        xrp.data, kernel.data, reg_kernel_weights.data, reg_kernels.data, d_next_xrp.data,
        d_xrp, d_reg_kernel_weights, d_reg_kernels)

    d_xrp = Variable(d_xrp)
    d_reg_kernel_weights = Variable(d_reg_kernel_weights)
    d_reg_kernels = Variable(d_reg_kernels)

    return d_xrp, None, d_reg_kernel_weights, d_reg_kernels


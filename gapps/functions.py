import inspect
import re

import numpy as np
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
  def forward(ctx, input, guide, filter):
    ctx.save_for_backward(input, guide, filter)

    bs, ci, h, w = input.shape
    co = filter.shape[0]

    assert guide.shape[0] == bs
    assert guide.shape[1] == h
    assert guide.shape[2] == w
    assert filter.shape[1] == ci

    output = input.new()
    output.resize_(bs, co, h, w);

    ops.bilateral_layer_forward(
        input, guide, filter, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input, guide, filter = ctx.saved_variables

    d_input = input.data.new()
    d_guide = guide.data.new()
    d_filter = filter.data.new()
    d_input.resize_as_(input.data)
    d_guide.resize_as_(guide.data)
    d_filter.resize_as_(filter.data)

    ops.bilateral_layer_backward(
        input.data, guide.data, filter.data, d_output.data,
        d_input, d_guide, d_filter)

    d_input = Variable(d_input)
    d_guide = Variable(d_guide)
    d_filter = Variable(d_filter)

    return d_input, d_guide, d_filter


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
  def forward(ctx, mosaick, selection_filters, green_filters, h_chroma, v_chroma, q_chroma):
    ctx.save_for_backward(mosaick, selection_filters, green_filters, h_chroma, v_chroma, q_chroma)

    output = mosaick.new()
    bs, ci, h, w = mosaick.shape
    assert ci == 1

    output.resize_(bs, 3, h, w)
    ops.learnable_demosaick_forward(
        mosaick.view(bs, h, w), selection_filters, green_filters, h_chroma, v_chroma, q_chroma, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    mosaick, selection_filters, green_filters, h_chroma, v_chroma, q_chroma = ctx.saved_variables

    d_mosaick = mosaick.data.new()
    d_mosaick.resize_as_(mosaick.data)
    d_sel_filts = selection_filters.data.new()
    d_sel_filts.resize_as_(selection_filters.data).zero_()
    d_green_filts = green_filters.data.new()
    d_green_filts.resize_as_(green_filters.data).zero_()
    d_h_chroma = h_chroma.data.new()
    d_h_chroma.resize_as_(h_chroma.data).zero_()
    d_v_chroma = v_chroma.data.new()
    d_v_chroma.resize_as_(v_chroma.data).zero_()
    d_q_chroma = q_chroma.data.new()
    d_q_chroma.resize_as_(q_chroma.data).zero_()

    bs, ci, h, w = mosaick.shape

    ops.learnable_demosaick_backward(
        mosaick.data.view(bs, h, w), selection_filters.data, green_filters.data,
        h_chroma.data, v_chroma.data, q_chroma.data,
        d_output.data,
        d_mosaick.view(bs, h, w), d_sel_filts, d_green_filts,
        d_h_chroma, d_v_chroma, d_q_chroma)

    d_mosaick = Variable(d_mosaick)
    d_sel_filts = Variable(d_sel_filts)
    d_green_filts = Variable(d_green_filts)
    d_h_chroma = Variable(d_h_chroma)
    d_v_chroma = Variable(d_v_chroma)
    d_q_chroma = Variable(d_q_chroma)

    return d_mosaick, d_sel_filts, d_green_filts, d_h_chroma, d_v_chroma, d_q_chroma

class DeconvCGInit(Function):
  """"""

  @staticmethod
  def forward(ctx, blurred, x0, kernel,
              data_kernel_weights, data_kernels,
              reg_kernel_weights, reg_kernels, reg_targets,
              w_data, w_reg):
    if any(ctx.needs_input_grad):
      ctx.save_for_backward(blurred, x0, kernel,
        data_kernel_weights, data_kernels,
        reg_kernel_weights, reg_kernels, reg_targets,
        w_data, w_reg)

    xrp = blurred.new()
    ci, h, w = blurred.shape

    xrp.resize_(3, ci, h, w)
    ops.deconv_cg_init_forward(
        blurred, x0, kernel,
        data_kernel_weights, data_kernels,
        reg_kernel_weights, reg_kernels, reg_targets,
        w_data, w_reg, xrp)

    return xrp

  @staticmethod
  def backward(ctx, d_xrp):
    blurred, x0, kernel, \
        data_kernel_weights, data_kernels, \
        reg_kernel_weights, reg_kernels, reg_targets, \
        w_data, w_reg = ctx.saved_variables

    d_x0 = x0.data.new()
    d_x0.resize_as_(x0.data)
    d_data_kernel_weights = data_kernel_weights.data.new()
    d_data_kernel_weights.resize_as_(data_kernel_weights.data)
    d_data_kernels = data_kernels.data.new()
    d_data_kernels.resize_as_(data_kernels.data)
    d_reg_kernel_weights = reg_kernel_weights.data.new()
    d_reg_kernel_weights.resize_as_(reg_kernel_weights.data)
    d_reg_kernels = reg_kernels.data.new()
    d_reg_kernels.resize_as_(reg_kernels.data)
    d_reg_targets = reg_targets.data.new()
    d_reg_targets.resize_as_(reg_targets.data)
    d_w_data = w_data.data.new()
    d_w_data.resize_as_(w_data.data)
    d_w_reg = w_reg.data.new()
    d_w_reg.resize_as_(w_reg.data)

    ci, h, w = blurred.shape

    ops.deconv_cg_init_backward(
        blurred.data, x0.data, kernel.data,
        data_kernel_weights.data, data_kernels.data,
        reg_kernel_weights.data, reg_kernels.data, reg_targets.data,
        w_data.data, w_reg.data, d_xrp.data,
        d_x0, d_data_kernel_weights, d_data_kernels,
        d_reg_kernel_weights, d_reg_kernels, d_reg_targets,
        d_w_data, d_w_reg)

    d_x0 = Variable(d_x0)
    d_data_kernel_weights = Variable(d_data_kernel_weights)
    d_data_kernels = Variable(d_data_kernels)
    d_reg_kernel_weights = Variable(d_reg_kernel_weights)
    d_reg_kernels = Variable(d_reg_kernels)
    d_reg_targets = Variable(d_reg_targets)
    d_w_data = Variable(d_w_data)
    d_w_reg = Variable(d_w_reg)

    return None, d_x0, None, \
           d_data_kernel_weights, d_data_kernels, \
           d_reg_kernel_weights, d_reg_kernels, d_reg_targets, \
           d_w_data, d_w_reg

class DeconvCGIter(Function):
  """"""

  @staticmethod
  def forward(ctx, xrp, kernel,
          data_kernel_weights, data_kernels,
          reg_kernel_weights, reg_kernels,
          w_data, w_reg):
    if any(ctx.needs_input_grad):
      ctx.save_for_backward(xrp, kernel,
                            data_kernel_weights, data_kernels,
                            reg_kernel_weights, reg_kernels,
                            w_data, w_reg)

    next_xrp = xrp.new()
    n, ci, h, w = xrp.shape
    assert n == 3
    assert ci == 3

    next_xrp.resize_(n, ci, h, w)
    ops.deconv_cg_iter_forward(
        xrp, kernel,
        data_kernel_weights, data_kernels,
        reg_kernel_weights, reg_kernels,
        w_data, w_reg, next_xrp)

    return next_xrp

  @staticmethod
  def backward(ctx, d_next_xrp):
    xrp, kernel, data_kernel_weights, data_kernels, \
      reg_kernel_weights, reg_kernels, w_data, w_reg = ctx.saved_variables

    d_xrp = xrp.data.new()
    d_xrp.resize_as_(xrp.data)
    d_data_kernel_weights = data_kernel_weights.data.new()
    d_data_kernel_weights.resize_as_(data_kernel_weights.data)
    d_data_kernels = data_kernels.data.new()
    d_data_kernels.resize_as_(data_kernels.data)
    d_reg_kernel_weights = reg_kernel_weights.data.new()
    d_reg_kernel_weights.resize_as_(reg_kernel_weights.data)
    d_reg_kernels = reg_kernels.data.new()
    d_reg_kernels.resize_as_(reg_kernels.data)
    d_w_data = w_data.data.new()
    d_w_data.resize_as_(w_data.data)
    d_w_reg = w_reg.data.new()
    d_w_reg.resize_as_(w_reg.data)

    ops.deconv_cg_iter_backward(
        xrp.data, kernel.data, 
        data_kernel_weights.data, data_kernels.data,
        reg_kernel_weights.data, reg_kernels.data,
        w_data.data, w_reg.data, d_next_xrp.data,
        d_xrp, 
        d_data_kernel_weights, d_data_kernels,
        d_reg_kernel_weights, d_reg_kernels, d_w_data, d_w_reg)

    d_xrp = Variable(d_xrp)
    d_data_kernel_weights = Variable(d_data_kernel_weights)
    d_data_kernels = Variable(d_data_kernels)
    d_reg_kernel_weights = Variable(d_reg_kernel_weights)
    d_reg_kernels = Variable(d_reg_kernels)
    d_w_data = Variable(d_w_data)
    d_w_reg = Variable(d_w_reg)

    return d_xrp, None, \
           d_data_kernel_weights, d_data_kernels, \
           d_reg_kernel_weights, d_reg_kernels, \
           d_w_data, d_w_reg

class DeconvCGWeight(Function):
  """"""

  @staticmethod
  def forward(ctx, blurred, current, reg_kernels, reg_targets, gmm_weights, 
              gmm_invvars):
    if any(ctx.needs_input_grad):
      ctx.save_for_backward(
          blurred, current, reg_kernels, reg_targets,
          gmm_weights, gmm_invvars)

    weights = blurred.new()
    ci, h, w = blurred.shape
    j, n = gmm_weights.shape
    assert ci == 3

    weights.resize_(n, ci, h, w)
    ops.deconv_cg_weight_forward(
        blurred, current, reg_kernels, reg_targets, gmm_weights, gmm_invvars, weights)

    return weights

  @staticmethod
  def backward(ctx, d_weights):
    blurred, current, reg_kernels, reg_targets, gmm_weights, gmm_invvars = ctx.saved_variables

    d_current = current.data.new()
    d_current.resize_as_(current.data)
    d_reg_kernels = reg_kernels.data.new()
    d_reg_kernels.resize_as_(reg_kernels.data)
    d_reg_targets = reg_targets.data.new()
    d_reg_targets.resize_as_(reg_targets.data)
    d_gmm_weights = gmm_weights.data.new()
    d_gmm_weights.resize_as_(gmm_weights.data)
    d_gmm_invvars = gmm_invvars.data.new()
    d_gmm_invvars.resize_as_(gmm_invvars.data)

    ops.deconv_cg_weight_backward(
        blurred.data, current.data, reg_kernels.data,
        reg_targets.data, gmm_weights.data, gmm_invvars.data, d_weights.data,
        d_current, d_reg_kernels, d_reg_targets, d_gmm_weights, d_gmm_invvars)

    d_current = Variable(d_current)
    d_reg_kernels = Variable(d_reg_kernels)
    d_reg_targets = Variable(d_reg_targets)
    d_gmm_weights = Variable(d_gmm_weights)
    d_gmm_invvars = Variable(d_gmm_invvars)

    return None, d_current, d_reg_kernels, d_reg_targets, d_gmm_weights, d_gmm_invvars

class BilateralGrid(Function):
  """"""

  @staticmethod
  def forward(ctx, input, filter_s, filter_r):
    if any(ctx.needs_input_grad):
      ctx.save_for_backward(input, filter_s, filter_r)

    c, h, w = input.shape

    output = input.new()
    output.resize_(c, h, w);

    ops.bilateral_grid_forward(
        input, filter_s, filter_r, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input, filter_s, filter_r = ctx.saved_variables

    d_input = input.data.new()
    d_input.resize_as_(input.data)
    d_filter_s = filter_s.data.new()
    d_filter_s.resize_as_(filter_s.data)
    d_filter_r = filter_r.data.new()
    d_filter_r.resize_as_(filter_r.data)

    ops.bilateral_grid_backward(
        input.data, filter_s.data, filter_r.data, d_output.data,
        d_input, d_filter_s, d_filter_r)

    d_input = Variable(d_input)
    d_filter_s = Variable(d_filter_s)
    d_filter_r = Variable(d_filter_r)

    return d_input, d_filter_s, d_filter_r

class DeconvPrior(Function):
  """"""

  @staticmethod
  def forward(ctx, f, reg_kernels, thresholds):
    if any(ctx.needs_input_grad):
      ctx.save_for_backward(f, reg_kernels, thresholds)

    weights = f.new()
    ci, h, w = f.shape
    n, _, _ = reg_kernels.shape
    assert ci == 3

    weights.resize_(n, ci, h, w)
    ops.deconv_prior_forward(
        f, reg_kernels, thresholds, weights)

    return weights

  @staticmethod
  def backward(ctx, d_weights):
    f, reg_kernels, thresholds = ctx.saved_variables

    d_f = f.data.new()
    d_f.resize_as_(f.data)
    d_reg_kernels = reg_kernels.data.new()
    d_reg_kernels.resize_as_(reg_kernels.data)
    d_thresholds = thresholds.data.new()
    d_thresholds.resize_as_(thresholds.data)

    ops.deconv_prior_backward(
        f.data, reg_kernels.data, thresholds.data,
        d_weights.data,
        d_f, d_reg_kernels, d_thresholds)

    d_f = Variable(d_f)
    d_reg_kernels = Variable(d_reg_kernels)
    d_thresholds = Variable(d_thresholds)

    return d_f, d_reg_kernels, d_thresholds

class NonLocalMeans(Function):
  """"""

  @staticmethod
  def forward(ctx, input, feature_filter, patch_filter, inv_sigma, search_radius):
    if any(ctx.needs_input_grad):
      ctx.save_for_backward(input, feature_filter, patch_filter, inv_sigma, search_radius)

    output = input.new()
    b, ci, h, w = input.shape
    assert ci == 3

    output.resize_(b, ci, h, w)
    ops.non_local_means_forward(
        input, feature_filter, patch_filter, inv_sigma[0], search_radius[0], output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input, feature_filter, patch_filter, inv_sigma, search_radius = ctx.saved_variables

    d_input = input.data.new()
    d_input.resize_as_(input.data)
    d_feature_filter = feature_filter.data.new()
    d_feature_filter.resize_as_(feature_filter.data)
    d_patch_filter = patch_filter.data.new()
    d_patch_filter.resize_as_(patch_filter.data)
    d_inv_sigma = inv_sigma.data.new()
    d_inv_sigma.resize_as_(inv_sigma.data)

    ops.non_local_means_backward(
        input.data, feature_filter.data, patch_filter.data, inv_sigma.data[0], search_radius[0],
        d_output.data,
        d_input, d_feature_filter, d_patch_filter, d_inv_sigma)

    d_input = Variable(d_input)
    d_feature_filter = Variable(d_feature_filter)
    d_patch_filter = Variable(d_patch_filter)
    d_inv_sigma = Variable(d_inv_sigma)

    return d_input, d_feature_filter, d_patch_filter, d_inv_sigma, None

class DeconvGrad(Function):
  """"""

  @staticmethod
  def forward(ctx, blurred, xk, kernel,
          data_kernel_weights, data_kernels, reg_kernel_weights, reg_kernels, reg_powers,
          reg_targets):
    if any(ctx.needs_input_grad):
      ctx.save_for_backward(blurred, xk, kernel,
        data_kernel_weights, data_kernels, reg_kernel_weights, reg_kernels, reg_powers, reg_targets)

    output = blurred.new()
    ci, h, w = blurred.shape
    assert ci == 3

    output.resize_(ci, h, w)
    ops.deconv_grad_forward(
        blurred, xk, kernel,
        data_kernel_weights, data_kernels,
        reg_kernel_weights, reg_kernels, reg_powers, reg_targets, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    blurred, xk, kernel, data_kernel_weights, data_kernels, \
        reg_kernel_weights, reg_kernels, reg_powers, reg_targets = ctx.saved_variables

    d_xk = xk.data.new()
    d_xk.resize_as_(xk.data)
    d_data_kernel_weights = data_kernel_weights.data.new()
    d_data_kernel_weights.resize_as_(data_kernel_weights.data)
    d_data_kernels = data_kernels.data.new()
    d_data_kernels.resize_as_(data_kernels.data)
    d_reg_kernel_weights = reg_kernel_weights.data.new()
    d_reg_kernel_weights.resize_as_(reg_kernel_weights.data)
    d_reg_kernels = reg_kernels.data.new()
    d_reg_kernels.resize_as_(reg_kernels.data)
    d_reg_powers = reg_powers.data.new()
    d_reg_powers.resize_as_(reg_powers.data)
    d_reg_targets = reg_targets.data.new()
    d_reg_targets.resize_as_(reg_targets.data)

    ops.deconv_grad_backward(
        blurred.data, xk.data, kernel.data, data_kernel_weights.data, data_kernels.data, reg_kernel_weights.data, reg_kernels.data, reg_powers.data, reg_targets.data,
        d_output.data,
        d_xk, d_data_kernel_weights, d_data_kernels, d_reg_kernel_weights, d_reg_kernels, d_reg_powers, d_reg_targets)

    d_xk = Variable(d_xk)
    d_data_kernel_weights = Variable(d_data_kernel_weights)
    d_data_kernels = Variable(d_data_kernels)
    d_reg_kernel_weights = Variable(d_reg_kernel_weights)
    d_reg_kernels = Variable(d_reg_kernels)
    d_reg_powers = Variable(d_reg_powers)
    d_reg_targets = Variable(d_reg_targets)

    return None, d_xk, None, d_data_kernel_weights, d_data_kernels, \
           d_reg_kernel_weights, d_reg_kernels, d_reg_powers, d_reg_targets

class DeconvAlpha(Function):
  """"""

  @staticmethod
  def forward(ctx, blurred, xk, kernel,
          data_kernel_weights, data_kernels, reg_kernel_weights, reg_kernels, reg_powers,
          reg_targets, direction):
    if any(ctx.needs_input_grad):
      ctx.save_for_backward(blurred, xk, kernel,
              data_kernel_weights, data_kernels, reg_kernel_weights, reg_kernels, reg_powers,
              reg_targets, direction)

    output = blurred.new()
    ci, h, w = blurred.shape
    assert ci == 3

    output.resize_(1)
    ops.deconv_alpha_forward(
        blurred, xk, kernel,
        data_kernel_weights, data_kernels,
        reg_kernel_weights, reg_kernels, reg_powers, reg_targets, direction, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    blurred, xk, kernel, data_kernel_weights, data_kernels, \
        reg_kernel_weights, reg_kernels, reg_powers, reg_targets, direction = ctx.saved_variables

    d_xk = xk.data.new()
    d_xk.resize_as_(xk.data)
    d_data_kernel_weights = data_kernel_weights.data.new()
    d_data_kernel_weights.resize_as_(data_kernel_weights.data)
    d_data_kernels = data_kernels.data.new()
    d_data_kernels.resize_as_(data_kernels.data)
    d_reg_kernel_weights = reg_kernel_weights.data.new()
    d_reg_kernel_weights.resize_as_(reg_kernel_weights.data)
    d_reg_kernels = reg_kernels.data.new()
    d_reg_kernels.resize_as_(reg_kernels.data)
    d_reg_powers = reg_powers.data.new()
    d_reg_powers.resize_as_(reg_powers.data)
    d_reg_targets = reg_targets.data.new()
    d_reg_targets.resize_as_(reg_targets.data)
    d_direction = direction.data.new()
    d_direction.resize_as_(direction.data)

    ops.deconv_alpha_backward(
        blurred.data, xk.data, kernel.data, data_kernel_weights.data, data_kernels.data, reg_kernel_weights.data, reg_kernels.data, reg_powers.data, reg_targets.data, direction.data,
        d_output.data,
        d_xk, d_data_kernel_weights, d_data_kernels, d_reg_kernel_weights, d_reg_kernels, d_reg_powers, d_reg_targets, d_direction)

    d_xk = Variable(d_xk)
    d_data_kernel_weights = Variable(d_data_kernel_weights)
    d_data_kernels = Variable(d_data_kernels)
    d_reg_kernel_weights = Variable(d_reg_kernel_weights)
    d_reg_kernels = Variable(d_reg_kernels)
    d_reg_powers = Variable(d_reg_powers)
    d_reg_targets = Variable(d_reg_targets)
    d_direction = Variable(d_direction)

    return None, d_xk, None, d_data_kernel_weights, d_data_kernels, \
           d_reg_kernel_weights, d_reg_kernels, d_reg_powers, d_reg_targets, d_direction

class SpatialTransformer(Function):
  """"""

  @staticmethod
  def forward(ctx, input, affine_mtx):
    ctx.save_for_backward(input, affine_mtx)

    output = input.new()
    bs, ci, h, w = input.shape

    assert affine_mtx.shape[0] == bs
    assert affine_mtx.shape[1] == 2
    assert affine_mtx.shape[2] == 3

    output.resize_(bs, ci, h, w)
    ops.spatial_transformer_forward(
        input, affine_mtx, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input, affine_mtx = ctx.saved_variables

    d_input = input.data.new()
    d_input.resize_as_(input.data)
    d_affine_mtx = affine_mtx.data.new()
    d_affine_mtx.resize_as_(affine_mtx.data)

    ops.spatial_transformer_backward(
        input.data, affine_mtx.data, d_output.data,
        d_input, d_affine_mtx)

    d_input = Variable(d_input)
    d_affine_mtx = Variable(d_affine_mtx)

    return d_input, d_affine_mtx


class BilinearResampling(Function):
  """"""

  @staticmethod
  def forward(ctx, input, warp):
    ctx.save_for_backward(input, warp)

    output = input.new()
    bs, ci, h, w = input.shape

    assert warp.shape[0] == bs
    assert warp.shape[1] == 2
    assert warp.shape[2] == h
    assert warp.shape[3] == w

    output.resize_(bs, ci, h, w)
    ops.bilinear_resampling_forward(
        input, warp, output)

    return output

  @staticmethod
  def backward(ctx, d_output):
    input, warp = ctx.saved_variables

    d_input = input.data.new()
    d_input.resize_as_(input.data)
    d_warp = warp.data.new()
    d_warp.resize_as_(warp.data)

    ops.bilinear_resampling_backward(
        input.data, warp.data, d_output.data,
        d_input, d_warp)

    d_input = Variable(d_input)
    d_warp = Variable(d_warp)

    return d_input, d_warp


class BurstDemosaicking(Function):
  """"""

  @staticmethod
  def forward(ctx, inputs, homographies, reconstructed, gradient_weight):
    ctx.save_for_backward(inputs, homographies, reconstructed, gradient_weight)

    loss = inputs.new()
    bs, h, w = inputs.shape

    reproj_error = inputs.new()

    assert homographies.shape[0] == bs
    assert homographies.shape[1] == 8

    assert reconstructed.shape[0] == 3
    # assert reconstructed.shape[1] == h
    # assert reconstructed.shape[2] == w

    loss.resize_(1)
    reproj_error.resize_(bs, h, w)
    ops.burst_demosaicking_forward(
        inputs, homographies, reconstructed, 
        gradient_weight, loss, reproj_error)

    return loss, reproj_error 

  @staticmethod
  def backward(ctx, d_loss, d_reproj_error):
    inputs, homographies, reconstructed, gradient_weight = ctx.saved_variables

    # d_confidence = confidence.data.new()
    # d_confidence.resize_as_(confidence.data)
    d_homographies = homographies.data.new()
    d_homographies.resize_as_(homographies.data)
    d_reconstructed = reconstructed.data.new()
    d_reconstructed.resize_as_(reconstructed.data)

    ops.burst_demosaicking_backward(
        inputs.data, homographies.data, reconstructed.data,
        gradient_weight.data, d_loss.data,
        d_homographies, d_reconstructed)

    # d_confidence = Variable(d_confidence)
    d_homographies = Variable(d_homographies)
    d_reconstructed = Variable(d_reconstructed)

    return None, d_homographies, d_reconstructed, None


class VGG(Function):
  """"""

  @staticmethod
  def forward(ctx, input, conv_weights, fc_weights, biases):
    ctx.save_for_backward(input, conv_weights, fc_weights, biases)

    bs, ci, h, w = input.shape

    n_out = fc_weights[-1].shape[0]

    output = input.new()
    output.resize_(bs, n_out)
    args = [input] + conv_weights + fc_weights + biases + [output]
    ops.vgg_forward(*args)

    return output
  #
  # @staticmethod
  # def backward(ctx, d_loss, d_reproj_error):
  #   inputs, homographies, reconstructed, gradient_weight = ctx.saved_variables
  #
  #   # d_confidence = confidence.data.new()
  #   # d_confidence.resize_as_(confidence.data)
  #   d_homographies = homographies.data.new()
  #   d_homographies.resize_as_(homographies.data)
  #   d_reconstructed = reconstructed.data.new()
  #   d_reconstructed.resize_as_(reconstructed.data)
  #
  #   ops.burst_demosaicking_backward(
  #       inputs.data, homographies.data, reconstructed.data,
  #       gradient_weight.data, d_loss.data,
  #       d_homographies, d_reconstructed)
  #
  #   # d_confidence = Variable(d_confidence)
  #   d_homographies = Variable(d_homographies)
  #   d_reconstructed = Variable(d_reconstructed)
  #
  #   return None, d_homographies, d_reconstructed, None

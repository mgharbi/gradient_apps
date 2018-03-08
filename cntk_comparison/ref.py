class BilateralLayerTorch(BilateralLayerBase):
  def __init__(self, ninputs, noutputs, kernel_size=3, use_bias=True):
    super(BilateralLayerTorch, self).__init__(
        ninputs, noutputs, kernel_size=kernel_size, use_bias=use_bias)
    self.sigma_s = 8
    self.sigma_r = 8

    self.conv = th.nn.Conv3d(
        ninputs, noutputs, self.kernel_size, bias=self.use_bias, 
        padding=self.kernel_size // 2)

    self.reset_params()

  def reset_params(self):
    if self.use_bias:
      self.conv.bias.data.zero_()

  def apply(self, input, guide):
    bs, ci, h, w = input.shape
    sigma_s = self.sigma_s
    sigma_r = self.sigma_r
    norm = 1.0/(sigma_s*sigma_s)

    guide = guide.unsqueeze(1)

    guide_pos = guide*sigma_r
    lower_bin = th.clamp(th.floor(guide_pos-0.5), min=0)
    upper_bin = th.clamp(lower_bin+1, max=sigma_r-1)
    weight = th.abs(guide_pos-0.5 - lower_bin)

    lower_bin = lower_bin.long()
    upper_bin = upper_bin.long()

    # Grid dimensions
    gw = w // sigma_s
    gh = h // sigma_s
    grid = input.new()
    grid.resize_(bs, ci, gh, gw, sigma_r)
    grid.zero_()

    # Splat
    batch_idx = th.from_numpy(np.arange(bs)).view(bs, 1, 1, 1)
    c_idx = th.from_numpy(np.arange(ci)).view(1, ci, 1, 1)
    h_idx = th.from_numpy(np.arange(h)).view(1, 1, h, 1) / sigma_s
    w_idx = th.from_numpy(np.arange(w)).view(1, 1, 1, w) / sigma_s
    grid[batch_idx, c_idx, h_idx, w_idx, lower_bin] += (1-weight)*norm*input
    grid[batch_idx, c_idx, h_idx, w_idx, upper_bin] += weight*norm*input

    # Conv3D
    grid = self.conv(grid)

    # Slice
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    gx = ((xx+0.5)/w) * gw
    gy = ((yy+0.5)/h) * gh
    gz = guide*sigma_r

    # Enclosing cell
    fx = np.floor(gx - 0.5).astype(np.int64);
    fy = np.floor(gy - 0.5).astype(np.int64);
    fz = th.clamp(th.floor(gz-0.5), min=0)
    cx = np.minimum(fx+1, gw-1);
    cy = np.minimum(fy+1, gh-1);
    cz = th.clamp(fz+1, max=sigma_r-1)

    # Trilerp weights
    wx = Variable(th.from_numpy((gx - 0.5 - fx).astype(np.float32)));
    wy = Variable(th.from_numpy((gy - 0.5 - fy).astype(np.float32)));
    wz = th.abs(gz-0.5 - fz)

    # Make indices broadcastable
    # fx = np.expand_dims(fx, 0)
    # fy = np.expand_dims(fy, 0)
    fz = fz.long()[:, 0].view(bs, 1, h, w)
    cz = cz.long()[:, 0].view(bs, 1, h, w)

    batch_idx = th.from_numpy(np.arange(bs)).view(bs, 1, 1, 1)
    c_idx = th.from_numpy(np.arange(ci)).view(1, ci, 1, 1)

    out = grid[batch_idx, c_idx, fy, fx, fz]*(1-wx)*(1-wy)*(1-wz) + \
          grid[batch_idx, c_idx, fy, fx, cz]*(1-wx)*(1-wy)*(  wz) + \
          grid[batch_idx, c_idx, cy, fx, fz]*(1-wx)*(  wy)*(1-wz) + \
          grid[batch_idx, c_idx, cy, fx, cz]*(1-wx)*(  wy)*(  wz) + \
          grid[batch_idx, c_idx, fy, cx, fz]*(  wx)*(1-wy)*(1-wz) + \
          grid[batch_idx, c_idx, fy, cx, cz]*(  wx)*(1-wy)*(  wz) + \
          grid[batch_idx, c_idx, cy, cx, fz]*(  wx)*(  wy)*(1-wz) + \
          grid[batch_idx, c_idx, cy, cx, cz]*(  wx)*(  wy)*(  wz)

    return out

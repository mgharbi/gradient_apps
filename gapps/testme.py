import torch  as th
import testop

print(dir(testop))

# TODO:
# test all types
# test CPU/GPU
# test correct computation
# test memory usage and release on GPU
# test correct GPU id
# test correct GPU stream

n = 4
c = 3
h = 32
w = 32

k = 3
co = 3

im = th.ones(n, c, h, w)
im[:, :, h//2, w//2] = 0

kernel = th.ones(co, c, k, k)
kernel[1, 0, 1, 1] = 1

out = th.zeros(n, co, h, w).double()

# print("running cpu")
# ret = testop.conv2d_forward(im, kernel, out)
# print(out.sum().item())

im = im.cuda()
kernel = kernel.cuda()
out = out.cuda()

out.zero_()

print("running gpu")
ret = testop.conv2d_forward_cuda(im, kernel, out)
print(out.sum().item())

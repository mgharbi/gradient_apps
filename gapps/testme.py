import torch  as th
import testop

def test_conv2d():
  print(dir(testop))

  # TODO:
  # test all types
  # test CPU/GPU
  # test correct computation
  # test memory usage and release on GPU
  # test correct GPU id
  # test correct GPU stream

  n = 1
  c = 3
  h = 5
  w = 5

  k = 3
  co = 3

  im = th.zeros(n, c, h, w)
  im[:, :, h//2, w//2] = 0

  kernel = th.zeros(co, c, k, k)
  kernel[1, 0, 1, 1] = 1

  out = th.zeros(n, co, h, w).double()

  ret = testop.conv2d_forward(im, kernel, out)

  print(out)

  # import os
  # import re
  # from torch.utils.cpp_extension import load
  #
  # abs_path = os.path.dirname(os.path.realpath(__file__))
  # build_dir = os.path.join(abs_path, "build")
  # src_dir = os.path.join(abs_path, "src")
  #
  # halide_dir = os.path.join(os.path.dirname(abs_path), "gradient-halide")
  # if halide_dir is None:
  #   raise ValueError("Please specify a HALIDE_DIR env variable.")
  #
  # if not os.path.exists(halide_dir):
  #   raise ValueError("Halide directory is invalid")
  #
  # # cuda_obj_re = re.compile(r".*_cuda.so")
  # # extra_objects = [f for f in os.listdir(build_dir) if os.path.splitext(f)[-1] == ".a"] 
  # # extra_objects += [f for f in os.listdir(build_dir) if cuda_obj_re.match(f)] 
  # # extra_objects = [os.path.join(build_dir, o) for o in extra_objects]
  #
  # # re_h = re.compile(r".*\.pytorch\.h")
  # # headers = [os.path.join(build_dir, f) for f in os.listdir(build_dir) if re_h.match(f)] 
  #
  # re_cc = re.compile(r".*\.pytorch\.cpp")
  # sources = [os.path.join(build_dir, f) for f in os.listdir(build_dir) if re_cc.match(f)] 
  #
  # print(sources)
  #
  # testop = load(
  #   name="testop", sources=sources, verbose=True,
  #   include_dirs=[os.path.join(abs_path, "build"), os.path.join(halide_dir, "include")]
  # )
  #
  # print(testop.forward)

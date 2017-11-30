import os
import re
from torch.utils.ffi import create_extension

abs_path = os.path.dirname(os.path.realpath(__file__))
build_dir = os.path.join(abs_path, "build")

halide_dir = os.getenv("HALIDE_DIR")
if halide_dir is None:
  raise ValueError("Please specify a HALIDE_DIR env variable.")

extra_objects = [f for f in os.listdir(build_dir) if os.path.splitext(f)[-1] == ".a"] 
extra_objects = [os.path.join(build_dir, o) for o in extra_objects]

re_h = re.compile(r".*\.pytorch\.h")
headers = [os.path.join(build_dir, f) for f in os.listdir(build_dir) if re_h.match(f)] 
re_cc = re.compile(r".*\.pytorch\.cpp")
sources = [os.path.join(build_dir, f) for f in os.listdir(build_dir) if re_cc.match(f)] 

exts = []

ffi = create_extension(
  name='_ext.operators',
  package=False,
  headers=headers,
  sources=sources,
  define_macros=[('WITH_CUDA', None)],
  language="c++",
  extra_objects=extra_objects,
  extra_compile_args=["-std=c++11"],
  relative_to=__file__,
  include_dirs=[os.path.join(abs_path, "build"), os.path.join(halide_dir, "include")],
  with_cuda=True
)

# ffi.cdef()
exts.append(ffi)

if __name__ == '__main__':
  for e in exts:
    e.build()

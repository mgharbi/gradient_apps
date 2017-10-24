import os
from torch.utils.ffi import create_extension

abs_path = os.path.dirname(os.path.realpath(__file__))
build_dir = os.path.join(abs_path, "build")

halide_dir = os.getenv("HALIDE_DIR")
if halide_dir is None:
  raise ValueError("Please specify a HALIDE_DIR env variable.")

# extra_objects = ["hl_operators.a"] 
extra_objects = [f for f in os.listdir(build_dir) if os.path.splitext(f)[-1] == ".a"] 
extra_objects = [os.path.join(build_dir, o) for o in extra_objects]

exts = []
exts.append(create_extension(
  name='_ext.operators',
  package=False,
  headers='src/operators.h',
  sources=['src/operators.cxx'],
  # language="c++",
  extra_objects=extra_objects,
  extra_compile_args=["-std=c++11"],
  relative_to=__file__,
  include_dirs=[os.path.join(abs_path, "build"), os.path.join(halide_dir, "include")],
))

if __name__ == '__main__':
  for e in exts:
    e.build()

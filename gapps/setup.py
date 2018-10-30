import os
import re
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

abs_path = os.path.dirname(os.path.realpath(__file__))
build_dir = os.path.join(abs_path, "build")
src_dir = os.path.join(abs_path, "src")

# Check Gradient Halide is available
halide_dir = os.path.join(os.path.dirname(abs_path), "gradient-halide")
if halide_dir is None:
  raise ValueError("Please specify a HALIDE_DIR env variable.")

if not os.path.exists(halide_dir):
  raise ValueError("Halide directory is invalid")

# Add all Halide generated torch wrapper
re_cc = re.compile(r".*\.pytorch\.cpp")
sources = [os.path.join(build_dir, f) for f in os.listdir(build_dir) if re_cc.match(f)] 
print("compiling custom modules with sources", sources)

# Add all Halide-generated libraries
extra_objects = [f for f in os.listdir(build_dir) if os.path.splitext(f)[-1] == ".a"] 
extra_objects = [os.path.join(build_dir, o) for o in extra_objects]

include_dirs = [os.path.join(abs_path, "build"), os.path.join(halide_dir, "include")]

setup(name="testop",
      verbose=True,
      ext_modules=[
        CppExtension("testop", sources, 
                     extra_objects=extra_objects,
                     extra_compile_args=["-std=c++11", "-stdlib=libc++"]
                     )],
      include_dirs=include_dirs,
      cmdclass={"build_ext": BuildExtension}
      )

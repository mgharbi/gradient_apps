import os
from torch.utils.ffi import create_extension

abs_path = os.path.dirname(os.path.realpath(__file__))

extra_objects = [os.path.join(abs_path, o) for o in extra_objects]

exts = []
exts.append(create_extension(
  name='_ext.operators',
  package=False,
  headers='src/operators.h',
  sources=['src/operators.cxx'],
  language="c++",
  extra_objects=extra_objects,
  extra_compile_args=["-std=c++11"],
  relative_to=__file__,
))

# exts.append(create_extension(
#   name='_ext.',
#   package=False,
#   headers='src/.h',
#   define_macros=[('WITH_CUDA', None)],
#   sources=['src/.cxx'],
#   language="c++",
#   extra_compile_args=["-std=c++11"],
#   relative_to=__file__,
#   extra_objects=[os.path.join(abs_path, 'build/kernels.so')],
#   with_cuda=True
# ))

if __name__ == '__main__':
  for e in exts:
    e.build()

import numpy as np
from scipy.ndimage import correlate

def make_mosaick(image):
  # Mosaick in G R format.
  #            B G
  mask = np.zeros_like(image)
  mask[0::2, 0::2, 1] = 1 
  mask[0::2, 1::2, 0] = 1 
  mask[1::2, 1::2, 1] = 1 
  mask[1::2, 0::2, 2] = 1 

  mosaick = np.sum(mask*image, axis=2)
  return mosaick

def sample_kernel(kernel_size):
  # For now return a box kernel
  return np.ones([kernel_size, kernel_size], dtype=np.float32) / float(kernel_size * kernel_size)

def make_blur(ref, kernel):
  blurred = correlate(ref, np.reshape(kernel, [kernel.shape[0], kernel.shape[1], 1]))
  return blurred

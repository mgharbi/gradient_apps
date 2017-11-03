import numpy as np

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

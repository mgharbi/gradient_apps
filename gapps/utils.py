import numpy as np
from scipy.ndimage import correlate
import math

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

def sample_trajectory(psf_size):
  # https://github.com/KupynOrest/DeblurGAN/blob/master/motion_blur/generate_trajectory.py
  expl = 0.005
  centripetal = 0.7 * np.random.uniform(0, 1)
  prob_big_shake = 0.2 * np.random.uniform(0, 1)
  gaussian_shake = 10 * np.random.uniform(0, 1)
  init_angle = 360 * np.random.uniform(0, 1)
  max_len = 60
  num_steps = 2000

  img_v0 = np.sin(np.deg2rad(init_angle))
  real_v0 = np.cos(np.deg2rad(init_angle))

  v0 = complex(real=real_v0, imag=img_v0)
  v = v0 * max_len / (num_steps - 1)

  x = np.array([complex(real=0, imag=0)] * (num_steps))

  for t in range(0, num_steps - 1):
    if np.random.uniform() < prob_big_shake * expl:
      next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
    else:
      next_direction = 0

    dv = next_direction + expl * (
      gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
        max_len / (num_steps - 1))

    v += dv
    v = (v / float(np.abs(v))) * (max_len / float((num_steps - 1)))
    x[t + 1] = x[t] + v

  # centere the motion
  border = 4.0 / psf_size
  x.real *= ((psf_size - border) / (np.max(x.real) - np.min(x.real)))
  x.imag *= ((psf_size - border) / (np.max(x.imag) - np.min(x.imag)))
  x += complex(real=-np.min(x.real) + 0.5 * border, imag=-np.min(x.imag) + 0.5 * border)

  return x

def sample_psf(psf_size):
  trajectory = sample_trajectory(psf_size)
  psf = np.zeros([psf_size, psf_size], dtype=np.float32)
  triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
  triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
  for t in range(len(trajectory)):
    m2 = int(np.minimum(psf_size - 2, np.maximum(0, np.math.floor(trajectory[t].real))))
    M2 = int(m2 + 1)
    m1 = int(np.minimum(psf_size - 2, np.maximum(0, np.math.floor(trajectory[t].imag))))
    M1 = int(m1 + 1)

    psf[m1, m2] += triangle_fun_prod(
      trajectory[t].real - m2, trajectory[t].imag - m1
    )
    psf[m1, M2] += triangle_fun_prod(
      trajectory[t].real - M2, trajectory[t].imag - m1
    )
    psf[M1, m2] += triangle_fun_prod(
      trajectory[t].real - m2, trajectory[t].imag - M1
    )
    psf[M1, M2] += triangle_fun_prod(
      trajectory[t].real - M2, trajectory[t].imag - M1
    )
  psf /= float(len(trajectory))
  psf /= np.sum(psf)
  return psf

  # For now return a box kernel
  # return np.ones([kernel_size, kernel_size], dtype=np.float32) / float(kernel_size * kernel_size)

def make_blur(ref, kernel, stddev = 0.01):
  blurred = correlate(ref, np.reshape(kernel, [kernel.shape[0], kernel.shape[1], 1]))
  blurred += np.random.normal(scale = stddev, size = blurred.shape)
  return np.clip(blurred, 0, 1).astype(np.float32)

def make_noisy(ref, stddev):
  return np.clip(ref + np.random.normal(scale = stddev, size = ref.shape), 0, 1).astype(np.float32)

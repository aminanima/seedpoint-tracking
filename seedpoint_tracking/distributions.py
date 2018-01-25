from __future__ import division

import numpy as np


def multivariate_gaussian(pos, mu, sigma):
  """
  Return the multivariate Gaussian distribution on array pos.
  pos is an array constructed by packing the meshed arrays of variables
  x_1, x_2, x_3, ..., x_k into its _last_ dimension.
  """
  n = mu.shape[0]
  sigma_det = np.linalg.det(sigma)
  sigma_inv = np.linalg.inv(sigma)
  N = np.sqrt((2 * np.pi) ** n * sigma_det)
  # This einsum call calculates (x-mu)T.sigma-1.(x-mu) in a vectorized
  # way across all the input variables.
  fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu)

  return np.exp(-fac / 2) / N

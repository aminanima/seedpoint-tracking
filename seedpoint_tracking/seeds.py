from __future__ import division

import numpy as np


def _sample_from_gaussian(width, height, c_x, c_y, sigma_exp):
  sigma = 2 ** sigma_exp
  # center of the Gaussian (in frame space) is the center of the GT bbox
  gaussian_mean = np.array([c_x, c_y])
  # covariance (diagonal) matrix defined by w and h of GT bbox
  gaussian_cov = np.array([[sigma * width ** 2, 0], [0, sigma * height ** 2]])
  # sample one seed from the bivariate Gaussian
  seedX, seedY = np.random.multivariate_normal(gaussian_mean, gaussian_cov, 1).T
  return seedX[0], seedY[0]


def seed_generator(mode, width, height, c_x, c_y, sigma_exp):
  """
  Generates seed-points given a mode and a bounding-box

  Parameters
  ----------
  mode: str
      Strategy used for seed generation. Only elongated_gaussian implemented for the moment
  width: int
      width of the bounding-box enclosing the original object
  height: int
      height of the bounding-box enclosing the original object
  c_x: float
      center-x coordinate of the bounding-box enclosing the original object
  c_y: float
      center-y coordinate of the bounding-box enclosing the original object
  sigma_exp: float
      Defines the variance of the Gaussian from which generating the seeds.
      The more negative, the closer to the center the samples will be.

  Returns
  -------
  seed_x: float
      x-coordinate of the seed
  seed_y: float
      y-coordinate of the seed

  """
  if mode == 'elongated_gaussian':
    seed_x, seed_y = _sample_from_gaussian(width, height, c_x, c_y, sigma_exp)
  elif mode == 'gaussian':
    raise NotImplementedError()
  else:
    raise ValueError('Unknown seed generator mode.')

  return seed_x, seed_y

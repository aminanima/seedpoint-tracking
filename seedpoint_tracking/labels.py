import numpy as np


# binary label of {-1, +1} used for SoftMarginLoss (logistic loss)
def binary(crop_side, batch_size, center=None, radius=1):
  label = (-1) * np.ones((batch_size, 1, crop_side, crop_side), np.float32)
  if center is None:
    center = np.asarray((np.floor(crop_side / 2), np.floor(crop_side / 2)), np.float32)
    center = np.tile(center, (batch_size, 1))

  for i in range(batch_size):
    label[i, :,
          int(round(center[i, 1]) - radius):int(round(center[i, 1]) + radius),
          int(round(center[i, 0]) - radius):int(round(center[i, 0]) + radius)] = 1

  return label


# can be used to re-balance the label (many more negative points)
def binary_weigths(crop_side, center=None, radius=1):
  n_pos = (radius * 2 + 1) ** 2
  n_neg = crop_side ** 2 - n_pos
  pos_weight = 0.5 / n_pos
  neg_weight = 0.5 / n_neg

  label = np.zeros((crop_side, crop_side), np.float32)

  if center is None:
    center = np.asarray((np.floor(crop_side / 2), np.floor(crop_side / 2)), np.float32)

  label[
      int(round(center[1]) - radius):int(round(center[1]) + radius),
      int(round(center[0]) - radius):int(round(center[0]) + radius)] = pos_weight

  label[label == 0] = neg_weight

  assert int(round(np.sum(label))) == 1
  return label


def gaussian(crop_side, batch_size, center=None, radius=8):
  label = np.zeros((batch_size, 1, crop_side, crop_side), np.float32)

  for i in range(batch_size):
    fwhm = radius * 2
    if center is not None:
      label[i, :, :, :] = make_gaussian(crop_side, fwhm, center[i, :])
    else:
      label[i, :, :, :] = make_gaussian(crop_side, fwhm)

    label[i, :, :, :] = label[i, :, :, :] / np.sum(label[i, :, :, :])

  assert int(round(np.sum(label))) == 1 * batch_size
  return label


def make_gaussian(size, fwhm, center=None):
  """ Make a square gaussian kernel.

  readout_crop_size is the length of a side of the square
  fwhm is full-width-half-maximum, which
  can be thought of as an effective radius.
  """

  x = np.arange(0, size, 1, float)
  y = x[:, np.newaxis]

  if center is None:
    x0 = y0 = np.floor(size / 2)
  else:
    x0 = center[0]
    y0 = center[1]

  return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

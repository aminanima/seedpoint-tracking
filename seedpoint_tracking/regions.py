from __future__ import division

import numpy as np


def region_to_bbox(region, center=True):
  """
  Return standard axis-aligned bounding-box.


  Parameters
  ----------
  region : np.ndarray
      bounding box annotation. It can be a axis-aligned or rotated bounding box (VOT format)
  center : bool
      if True return the middle of the bounding box instead of its top-left corner

  Returns
  ----------
  np.ndarray
      axis-aligned bounding-box with format
      <leftX, topY, width, height> or <centerX, centerY, width, height>

  """

  n = len(region)
  assert n == 4 or n == 8, 'GT region format is invalid, should have 4 or 8 entries.'

  if n == 4:
    return _rect(region, center)
  else:
    return _rotated_rect(region, center)


def _rect(region, center):
  """Return axis-aligned bounding-box as it is or with center coordinate"""
  if center:
    leftX, topY, width, height = region
    centerX = leftX + width / 2
    centerY = topY + height / 2
    return centerX, centerY, width, height
  else:
    return region


def _rotated_rect(region, center):
  """Find the axis-aligned bounding-box with the same area and center of the rotated bounding-box"""
  x1 = np.min(region[::2])
  x2 = np.max(region[::2])
  y1 = np.min(region[1::2])
  y2 = np.max(region[1::2])
  centerX = (x1 + x2) / 2
  centerY = (y1 + y2) / 2
  A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
  A2 = (x2 - x1) * (y2 - y1)
  s = np.sqrt(A1 / A2)
  width = s * (x2 - x1)
  height = s * (y2 - y1)

  if center:
    return centerX, centerY, width, height
  else:
    return centerX - width / 2, centerY - height / 2, width, height

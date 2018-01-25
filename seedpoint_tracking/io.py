"""File io functions"""
from PIL import Image
import numpy as np


def open_image(image_path):
  """
  Wrapper around Image.open

  Parameters
  ----------
  file_uri : str
      URI to file. Can be local path, hdfs URI or URL to file on web/blobstore (http).

  Returns
  -------
  np.ndarray
      Image data in BGR format. shape = (height, width, channels)
  """

  img = Image.open(image_path)
  data_rgb = np.array(img)
  data_bgr = np.ascontiguousarray(data_rgb[:, :, 2::-1])  # rgb to bgr

  return data_bgr

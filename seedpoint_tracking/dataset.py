from __future__ import division

import logging
from os.path import join

from PIL import Image
import numpy as np
from pandas import DataFrame
import requests
from torch.utils.data import Dataset


class SeedpointTrackingDataset(Dataset):
  """Seedpoint-tracking dataset."""

  def __init__(self, root_folder, csv_path, transform=None):
    self.root_folder = root_folder
    self.transform = transform
    self.csv_content = DataFrame.from_csv(csv_path)

  def __len__(self):
    return len(self.csv_content)

  def __getitem__(self, idx):
    """
    Returns:
        ``sample`` (tuple): content of CSV row
    """

    # TODO Check the reading of csv data is correct
    csv_row = self.csv_content.iloc[idx]

    try:
      img = Image.open(join(self.root_folder, csv_row.filename))
    except (requests.RequestException, ValueError) as e:
      logger = logging.getLogger("seedpoint_tracking")
      logger.warning("An error occured while opening an image: {}".format(type(e)))
      return None
    seed = np.asarray([csv_row.seedX, csv_row.seedY], np.float32)
    center = np.asarray([csv_row.cX, csv_row.cY], np.float32)
    # dictionary support included for torch.__version__ >0.11
    # sample = {'img': img, 'center': center, 'seed': seed}
    sample = img, center, seed

    if self.transform is not None:
      sample = self.transform(sample)

    return sample

from __future__ import division

import logging
from os.path import join
import pickle
import random
import re

from PIL import Image
import numpy as np
from pandas import DataFrame
import requests
from torch.utils.data import Dataset


class TrackingDataset(Dataset):
  """Seedpoint-tracking dataset."""

  def __init__(self, root_folder, dataset_path, exemplar_max_distance,
               video_dict_path, frame_dict_path, transform=None):
    self.root_folder = root_folder
    self.transform = transform
    self.dataset = DataFrame.from_csv(dataset_path)
    # maximum 'distance' in frames from which to sample the exemplar
    self.exemplar_max_distance = exemplar_max_distance

    with open(video_dict_path, 'rb') as handle:
      self.video_dict = pickle.load(handle)

    with open(frame_dict_path, 'rb') as handle:
      self.frame_dict = pickle.load(handle)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    """
    Returns:
        ``sample`` (tuple): content of CSV row
    """

    # TODO Check the reading of csv data is correct
    csv_row = self.dataset.iloc[idx]
    exemplar_filename = csv_row.filename
    # used for ilsvrc, where there can be multiple object per frame
    exemplar_track_id = csv_row.trackID
    # filename and bbox for search crop
    search_filename, search_bbox = self._get_random_search_area(
      exemplar_filename, exemplar_track_id)

    try:
      exemplar_img = Image.open(join(self.root_folder, exemplar_filename))
    except (requests.RequestException, ValueError) as e:
      logger = logging.getLogger("seedpoint_tracking")
      logger.warning("An error occured while opening an image: {}".format(type(e)))
      return None

    try:
      search_img = Image.open(join(self.root_folder, search_filename))
    except (requests.RequestException, ValueError) as e:
      logger = logging.getLogger("seedpoint_tracking")
      logger.warning("An error occured while opening search image: {}".format(type(e)))
      return None

    # otb contains some grayscale videos
    if exemplar_img.layers == 1:
      exemplar_img = exemplar_img.convert('RGB')
      search_img = search_img.convert('RGB')

    exemplar_center = np.asarray((csv_row.cX, csv_row.cY), np.float32)
    exemplar_bbox = np.asarray(
        (csv_row.leftX,
         csv_row.topY,
         csv_row.rightX,
         csv_row.bottomY),
        np.float32)
    # readout_crop_size of the object in the crops
    exemplar_obj_size = self._size_from_bbox(exemplar_bbox)
    search_obj_size = self._size_from_bbox(search_bbox)
    search_center = np.asarray(
        (search_bbox[0] +
         search_obj_size[0] /
         2,
         search_bbox[1] +
         search_obj_size[1] /
         2),
        np.float32)
    # read the seed annotations from dataset
    seed = np.asarray((csv_row.seedX, csv_row.seedY), np.float32)
    # pack everything into a sample to pass to multiple transforms
    sample = exemplar_img, search_img, exemplar_obj_size, exemplar_center, search_center, seed

    if self.transform is not None:
      sample = self.transform(sample)

    return sample


def _get_random_search_area(self, exemplar_filename, exemplar_track_id):

  extension = exemplar_filename.split('.')[-1]
  video_path = re.sub('/[0-9]+\.' + extension, '', exemplar_filename)
  exemplar_filename = exemplar_filename.split('/')[-1].replace('.' + extension, '')
  n_digits_in_frame = len(exemplar_filename)
  exemplar_frame_int = int(exemplar_filename)
  video_id = video_path.replace('/', '-') + '-track' + exemplar_track_id
  candidate_frames = self.video_dict[video_id]
  suitable_frames = [f for f in candidate_frames if abs(
    f - exemplar_frame_int) <= self.exemplar_max_distance]
  # workaround needed to avoid nan entries in pickled dictionary
  found = False
  while not found:
    search_frame_int = random.choice(suitable_frames)
    search_frame_id = 'frame%08d' % search_frame_int
    search_bbox = self.frame_dict[video_id + '-' + search_frame_id]
    if not np.isnan(search_bbox).any():
      found = True

  search_frame_path = video_path + '/' + str(
    search_frame_int).zfill(n_digits_in_frame) + '.' + extension

  return search_frame_path, search_bbox


def _size_from_bbox(self, bbox):
  return np.asarray((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), np.float32)

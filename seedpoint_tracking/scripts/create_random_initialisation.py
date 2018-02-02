"""
For each video in the dataset, creates a new initialization with a "guessed" bounding box, its
center is sampled from the Gaussian by the input sigma, its size depends only on the frame area.
"""

from __future__ import division

import argparse
from os import listdir
from os.path import isfile, join

import numpy as np
from seedpoint_tracking.regions import region_to_bbox
from seedpoint_tracking.video import video_infos


VIDEO_ROOT = '../../data/tracking'
VIDEO_DATASET = 'validation'
DATASET_FOLDER = join(VIDEO_ROOT, VIDEO_DATASET)
# avg w/h of targets from OTB-100
AVG_ASPECT_RATIO = 0.730
# avg sqrt(target_area/frame_area) from OTB-100
AVG_FRAME_TO_TARGET = 0.173
EPSILON = 0.001


def _print_to_file(bb, file):
  bb = np.round(bb, decimals=1)
  bounding_box = ','.join(map(str, bb))
  print(bounding_box)
  bounding_box = bounding_box + '\n'
  file.write(bounding_box)


def create_random_initialisation(sigma, init_file):
  """Used to create dummy initialization for preliminary experiment.

  Takes groundtruth of videos in VIDEO_DATASET and generates initialization by sampling
  bounding-box centers on an elongated gaussian shaped as the groundtruth. AVG_FRAME_TO_TARGET and
  AVG_ASPECT_RATIO define the bounding-box size and have been computed on OTB-100 dataset.
  """
  videos_list = sorted([v for v in listdir(DATASET_FOLDER)])
  for i in range(len(videos_list)):
    video_folder = join(DATASET_FOLDER, videos_list[i])
    gt, frame_size, frame_name_list = video_infos(video_folder)
    init_file_path = join(video_folder, init_file)
    # make sure file does not exist already, then open it
    assert not isfile(init_file_path), 'Initialization file already exists'

    with open(init_file_path, 'a') as f:
      print('Writing in ' + init_file_path)
      for j in range(len(frame_name_list)):
        bbox = region_to_bbox(gt[j, :], center=False)
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        ground_truth_width, ground_truth_height = bbox[2], bbox[3]
        # center of the Gaussian (in frame space) is the center of the GT bbox
        gaussian_mean = np.array([cx, cy])
        # covariance (diagonal) matrix defined by w and h of GT bbox
        gaussian_cov = np.array(
          [[sigma * ground_truth_width ** 2, 0], [0, sigma * ground_truth_height ** 2]])
        # sample one seed from the bivariate Gaussian
        seed_x, seed_y = np.random.multivariate_normal(gaussian_mean, gaussian_cov, 1).T
        seed_x, seed_y = seed_x[0], seed_y[0]
        # dummy guess of target area based on frame area (avg from OTB-100)
        guessed_target_area = np.power(np.sqrt(np.prod(frame_size)) * AVG_FRAME_TO_TARGET, 2)
        # dummy guess of target w and h based on avg from OTB-100
        guessed_height = np.sqrt(guessed_target_area / AVG_ASPECT_RATIO)
        guessed_width = AVG_ASPECT_RATIO * guessed_height
        assert guessed_width / guessed_height <= AVG_ASPECT_RATIO + EPSILON
        assert guessed_width / guessed_height >= AVG_ASPECT_RATIO - EPSILON
        assert np.sqrt(
          (guessed_width * guessed_height) / np.prod(frame_size)) <= AVG_FRAME_TO_TARGET + EPSILON
        assert np.sqrt(
          (guessed_width * guessed_height) / np.prod(frame_size)) >= AVG_FRAME_TO_TARGET - EPSILON
        guessed_bbox = (seed_x - guessed_width / 2, seed_y - guessed_height / 2, guessed_width,
                        guessed_height)
        # append to file
        _print_to_file(guessed_bbox, f)
      f.close()
      print()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--sigma', type=float,
                      help='Defines the Gaussian centered on the ground-truth bbox center')
  parser.add_argument('--init_file', type=str,
                      help='Suffix of the name of the new initialization file')

  options = parser.parse_args()
  create_random_initialisation(**vars(options))

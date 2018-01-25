from __future__ import division

import argparse
from json import loads as json_loads
from os import mkdir
from os.path import join as join_path

from PIL import Image
import numpy as np
from pandas import DataFrame
from seedpoint_tracking.video import save_frame_with_annotation


def save_frames_with_annotators_output(seeds_per_image, gt_path, annotators_csv_path, frames_dir,
                                       save_dir):
  """
  Useful to show the output of the annotators vs the ground-truth center.

  Parameters
  ----------
  seeds_per_image: int
      Number of seeds per image.
  gt_path: str
      Full path of CSV file with ground truth bounding box annotation.
  annotators_csv_path: str
      Full path of csv with annotation.
  frames_dir: str
      Full path of folder where original frames can be found.
  save_dir: str
      Full path of the directory in which the frames should be saved.

  Returns
  -------

  """

  # load ground-truth for annotations in dictionary
  gt_center = dict()
  gt_area = dict()
  gt_content = DataFrame.from_csv(gt_path)
  for i in range(len(gt_content)):
    gt_row = gt_content[i]
    gt_center[gt_row.filename] = (
      float(gt_row.leftX) + float(gt_row.width) / 2,
      float(gt_row.topY) + float(gt_row.height) / 2)

    gt_area[gt_row.filename] = np.sqrt(float(gt_row.width) * float(gt_row.height))

  # read output of annotators
  csv_content = DataFrame.from_csv(annotators_csv_path)
  seeds = []
  frame_name_old = ''
  frames = 0
  cum_center_dist = 0

  for i in range(len(csv_content)):

    print('Parsiong annotators CSV, line ' + str(i))
    csv_row = csv_content.iloc[i]
    geo_json = json_loads(csv_row.ia)
    frame_name = csv_row.filename.rsplit('/', 1)[1]
    # Code assumes that in the csv produced by annotators multiple annotations for the same image
    # are consecutive
    if seeds:
      assert frame_name == frame_name_old
    src_dataset = frame_name.rsplit('-')[0]
    # fix filename format to be compatible with ilsvrc (which also has trackid)
    if src_dataset == 'otb100' or src_dataset == 'pool574' or src_dataset == 'uav123':
      frame_name_fix = frame_name.rsplit('.')[0] + '-0.jpg'
    else:
      frame_name_fix = frame_name

    # check if annotator reported the point
    if geo_json[0]['paths']:
      # seed point is saved as a cross on the actual seed
      cross = geo_json[0]['paths'][0][1]['segments']
      img = Image.open(join_path(frames_dir, frame_name))
      seed = (cross[1][0][0], cross[0][0][1])
      seeds.append(seed)
      frames += 1
      # distance seed to center normalized by the sqrt of the ground-truth bounding-box area
      cum_center_dist += np.linalg.norm(
        np.asarray(seed) - np.asarray(gt_center[frame_name_fix])) / gt_area[frame_name_fix]

    else:
      print('WARNING: ' + frame_name + ' is missing one annotation.')

    if (i + 1) % seeds_per_image == 0:
      # when all the seeds for the current image are read visualize them on it and re-init the list
      save_frame_with_annotation(img, join_path(save_dir, frame_name),
                                 seeds=seeds, center=gt_center[frame_name_fix])
      seeds = []

    frame_name_old = frame_name

  return np.round(cum_center_dist / frames, 4), frames


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--seeds_per_image', type=int, default=5,
                      help='Number of seed per image.')

  parser.add_argument('--gt_path', type=str,
                      default='../../../data/tracking/spt_annotations_july/annotationCenterGT.csv',
                      help='Full path of CSV file with ground truth bounding box annotation.')

  parser.add_argument('--annotators_csv_path', type=str,
                      default='../../../data/tracking/spt_annotations_july/'
                              'HCOMP-1179-results/batch_10_results.csv',
                      help='Full path of csv with annotation.')

  parser.add_argument('--frames_dir', type=str,
                      default='../../../data/tracking/spt_annotations_july/'
                              'frames_for_annotators',
                      help='Full path of folder with original frames.')

  parser.add_argument('--save_dir', type=str,
                      default='../../../data/tracking/spt_annotations_july/'
                              'HCOMP-1179-results/batch_10_frames',
                      help='Full path of the directory in which to save the frames.')

  args = parser.parse_args()

  mkdir(args.save_dir)

  avg_dist_from_center, num_valid_annotations = save_frames_with_annotators_output(**vars(args))

  print('\nOutput from ' + args.annotators_csv_path + ' has average normalized distance of ' + str(
    avg_dist_from_center) + ' over ' + str(num_valid_annotations) + ' valid annotations.')

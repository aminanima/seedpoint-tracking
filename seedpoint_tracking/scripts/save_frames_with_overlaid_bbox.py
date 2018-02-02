from __future__ import division

import argparse
import csv
from os import mkdir
from os.path import join as join_path

from PIL import Image
import numpy as np
from pandas import DataFrame
from seedpoint_tracking.video import save_frame_with_annotation


def _dilate_bbox(bbox, percentage, frame_size):
  """ Dilate the bounding-box so that it does not obstruct the boundaries of the object. """
  left_x = max(bbox[0] - bbox[2] * percentage / 2, 0)
  width = min(bbox[2] + bbox[2] * percentage, frame_size[0])
  top_y = max(bbox[1] - bbox[3] * percentage / 2, 0)
  height = min(bbox[3] + bbox[3] * percentage, frame_size[1])
  return [left_x, top_y, width, height]


def save_frames_with_overlaid_bbox(root_folder, dataset_csv, dilate_bbox, out_dir, out_gt):
  """
  Prepare the frames for the annotators.
  """

  csv_content = DataFrame.from_csv(dataset_csv)
  gt_data = [['filename', 'leftX', 'topY', 'width', 'height']]

  for index, csv_row in csv_content:
    bbox = np.asarray((csv_row.left_x, csv_row.top_y, csv_row.width, csv_row.height), np.float32)
    frame_size = np.asarray((csv_row.frameWidth, csv_row.frameHeight), np.float32)
    dil_bbox = _dilate_bbox(bbox, dilate_bbox, frame_size)
    img = Image.open(join_path(root_folder, csv_row.filename))
    # replace '/' with '-' in the file names
    new_filename = csv_row.filename.replace('/', '-')
    new_filename = new_filename.split('.')[0] + '-' + csv_row.trackID + '.jpg'
    # convert to x0,y0,x1,y1 format for ImageDraw and save frame
    bbox_coords = [dil_bbox[0], dil_bbox[1], dil_bbox[0] + dil_bbox[2], dil_bbox[1] + dil_bbox[3]]
    save_frame_with_annotation(img, join_path(out_dir, new_filename), bbox=bbox_coords)
    # save new filename and ground truth center for easier access for post-annotation visualization
    gt_data.append([new_filename, csv_row.leftX, csv_row.topY, csv_row.width, csv_row.height])
    print('Done with ' + new_filename)

  with open(out_gt, 'w', newline='') as fp:
    wr = csv.writer(fp, dialect='unix', quoting=csv.QUOTE_NONE)
    wr.writerows(gt_data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--root_folder', type=str,
                      default='../../../../external_hd/ILSVRC2015/Data/VID',
                      help='Root path of the video frames')

  parser.add_argument('--dataset_csv', type=str,
                      default='../../../data/tracking/'
                              'dataset.ilsvrc_train.forAnnotators.eGauss-5.csv',
                      help='Full path of dataset CSV')

  parser.add_argument('--dilate_bbox', type=float, default=0.2,
                      help='Percentage of dilation of bbox wrt ground truth.')

  parser.add_argument('--out_dir', type=str,
                      default='../../../data/tracking/ilsvrc_train_frames_for_annotators',
                      help='Path of the directory in which the frames with overlay should be saved')

  parser.add_argument('--out_gt', type=str,
                      default='../../../data/tracking/spt_annotations_july/'
                              'ilsvrc_train.annotationCenterGT.csv',
                      help='Path of the file with ground truth center '
                           'for the files given to the annotators.')

  args = parser.parse_args()

  mkdir(args.out_dir)

  save_frames_with_overlaid_bbox(**vars(args))

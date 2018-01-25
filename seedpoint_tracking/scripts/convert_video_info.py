from __future__ import division

import argparse
from os import listdir
from os.path import join

import numpy as np
from seedpoint_tracking.video import video_infos


VIDEO_ROOT = '../../data/tracking'
VIDEO_DATASET = 'validation574'
DATASET_FOLDER = join(VIDEO_ROOT, VIDEO_DATASET)


def convert_video_info(source_dataset):
  """
  Converts groundtruth files from <x1,y1,x2,y2> to <x1,y1,w,h> format
  """
  videos_list = sorted([v for v in listdir(DATASET_FOLDER) if v.startswith(source_dataset)])
  for i in range(len(videos_list)):
    video_folder = join(DATASET_FOLDER, videos_list[i])
    print('Converting ground truth of ' + videos_list[i])
    original_video_info, _, _ = video_infos(video_folder)
    converted_video_info = np.empty_like(original_video_info)
    n_frames = original_video_info.shape[0]
    for j in range(n_frames):
      bbox_in = original_video_info[j, :]
      converted_video_info[j, :] = np.asarray(
        (bbox_in[0], bbox_in[1], bbox_in[2] - bbox_in[0], bbox_in[3] - bbox_in[1]))
    file_out = join(video_folder, 'groundtruth_rect.txt')
    np.savetxt(file_out, converted_video_info, delimiter=',')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dataset', type=str,
                      help='Prefix of videos folder, indicates the source dataset.')

  options = parser.parse_args()
  convert_video_info(**vars(options))

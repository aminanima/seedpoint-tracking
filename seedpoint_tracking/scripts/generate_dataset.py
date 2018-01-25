from __future__ import division

import argparse
import csv
from os import listdir
from os.path import join
import xml.etree.ElementTree as ET

import numpy as np
from seedpoint_tracking.regions import region_to_bbox
from seedpoint_tracking.seeds import seed_generator
from seedpoint_tracking.video import video_infos


FRAMES_TO_SKIP_AFTER_START = 10


def _generate_data_default(data, root_folder, dataset_folder, source_sequences, seed_mode,
                           sigma_exp):
  seed_mode_id = seed_mode + str(sigma_exp)

  videos_list = sorted([v for v in listdir(join(root_folder, dataset_folder)) if
                        v.startswith(source_sequences)])

  for i in range(len(videos_list)):
    video_folder = join(dataset_folder, videos_list[i])
    print('Saving: ' + str(i) + ' ' + videos_list[i])
    gt, frame_size, frame_name_list = video_infos(join(root_folder, video_folder))

    # define 6 subsequences according to video length
    sub_starts = np.round(np.linspace(0, len(frame_name_list), 6 + 1))[:-1]

    for j in range(len(frame_name_list)):
      is_sequence_start = j == 0
      is_subsequence_start = j in sub_starts

      left_x, top_y, width, height = np.asarray(region_to_bbox(gt[j, :], center=False))
      right_x = left_x + width
      bottom_y = top_y + height
      c_x = left_x + width / 2
      c_y = top_y + height / 2
      seed_x, seed_y = seed_generator(seed_mode, width, height, c_x, c_y, sigma_exp)
      tmp = left_x, top_y, right_x, bottom_y, c_x, c_y, width, height, seed_x, seed_y
      left_x, top_y, right_x, bottom_y, c_x, c_y, width, height, seed_x, seed_y = np.round(tmp, 2)

      # Avoid printing rootDir from csv to make it machine-independent
      frame_name = frame_name_list[j].replace(root_folder + '/', '')
      print('Processing ' + frame_name)

      csv_line = [is_sequence_start, is_subsequence_start, frame_name, frame_size[0], frame_size[1],
                  left_x, top_y, right_x, bottom_y,
                  c_x, c_y,
                  width, height,
                  # trackID, classID, occluded: fields for ilsvrc compatibility
                  0, "NA", "",
                  seed_x, seed_y, seed_mode_id]

      data.append(csv_line)

  return data


def _generate_data_ilsvrc(data, root_folder, dataset_folder, seed_mode, sigma_exp):
  seed_mode_id = seed_mode + str(sigma_exp)

  folder_list = sorted([f for f in listdir(join(root_folder, dataset_folder))])

  # For convenience, we only consider first frames for annotation purpose in ilsvrc15
  is_subsequence_start = False

  for i in range(len(folder_list)):

    folder = join(root_folder, dataset_folder, folder_list[i])
    video_list = sorted([v for v in listdir(folder)])

    for j in range(len(video_list)):

      video = join(folder, video_list[j])
      frame_annotation_list = sorted([a for a in listdir(video) if a.endswith('.xml')])

      # initialize empty dictionary of track_ids
      track_ids = dict()

      for k in range(len(frame_annotation_list)):

        frame_annotation = join(video, frame_annotation_list[k])

        xml_tree = ET.parse(frame_annotation)
        xml_root = xml_tree.getroot()
        frame_name = join(dataset_folder, folder_list[i], video_list[j],
                          xml_root.find('filename').text + '.JPEG')
        frame_width = int(xml_root.find('size').find('width').text)
        frame_height = int(xml_root.find('size').find('height').text)

        print('Processing ' + frame_name)

        for xml_object in xml_root.findall('object'):

          # If it's the first time in a video that an object appears, we consider it as new sequence

          # We let 10 frames pass to reduce likelihood that object is cut because it is moving into
          # the scene

          track_id = xml_object.find('trackid').text
          if track_id in track_ids.keys():
            track_ids[track_id] += 1
          else:
            track_ids[track_id] = 0

          if track_ids[track_id] == FRAMES_TO_SKIP_AFTER_START:
            is_sequence_start = True
          else:
            is_sequence_start = False

          class_id = xml_object.find('name').text
          occluded = bool(xml_object.find('occluded').text)

          xml_bbox = xml_object.find('bndbox')
          right_x = np.float32(xml_bbox.find('xmax').text)
          bottom_y = np.float32(xml_bbox.find('ymax').text)
          left_x = np.float32(xml_bbox.find('xmin').text)
          top_y = np.float32(xml_bbox.find('ymin').text)
          width = right_x - left_x
          height = bottom_y - top_y
          c_x = left_x + width / 2
          c_y = top_y + height / 2

          seed_x, seed_y = seed_generator(seed_mode, width, height, c_x, c_y, sigma_exp)
          tmp = left_x, top_y, right_x, bottom_y, c_x, c_y, width, height, seed_x, seed_y
          tmp = np.round(tmp, 2).astype(np.float32)
          left_x, top_y, right_x, bottom_y, c_x, c_y, width, height, seed_x, seed_y = tmp

          csv_line = [is_sequence_start, is_subsequence_start,
                      frame_name, frame_width, frame_height,
                      left_x, top_y, right_x, bottom_y,
                      c_x, c_y,
                      width, height,
                      track_id, class_id, occluded,
                      seed_x, seed_y, seed_mode_id]

          data.append(csv_line)

  return data


def generate_dataset(root_folder, dataset_folder, source_sequences, csv_out_name, seed_mode,
                     sigma_exp, ilsvrc):
  """
  Prepare .csv with all groundtruth information, one frame per line.
  The code being run is significantly different in case of ilsvrc15vid dataset, which a) stores info
  as xml and
  b) has mulitple annotation per frame.
  """

  # Initialize the CSV file and save headers as first row
  data = [['isSequenceStart', 'isSubsequenceStart',
           'filename', 'frameWidth', 'frameHeight',
           'leftX', 'topY', 'rightX', 'bottomY',
           'cX', 'cY',
           'width', 'height',
           # !=0 in case of multiple object per frame (only in ilsvrc15vid)
           'trackID',
           # data available only for ilsvrc, empty string otherwise
           'classID', 'occluded',
           # seed infos are not inherent properties of the dataset, but for convenience we store it
           # in the csv
           'seedX', 'seedY', 'seedMode']]

  if not ilsvrc:
    data = _generate_data_default(data, root_folder, dataset_folder, source_sequences, seed_mode,
                                  sigma_exp)
  else:
    data = _generate_data_ilsvrc(data, root_folder, dataset_folder, seed_mode, sigma_exp)

  with open(csv_out_name, 'w', newline='') as fp:
    wr = csv.writer(fp, dialect='unix', quoting=csv.QUOTE_NONE)
    wr.writerows(data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--root_folder', type=str,
                      default='../../../../external_hd/ILSVRC2015/Annotations/VID')

  parser.add_argument('--dataset_folder', type=str, default='train')

  parser.add_argument('--source_sequences', type=str, default='',
                      help='Sequences prefix (indicates original dataset')

  parser.add_argument('--csv_out_name', type=str,
                      default='../../../data/tracking/dataset.ilsvrc_train.eGauss-5.csv',
                      help='Full path of the file in which the dataset should be saved (as CSV).')

  parser.add_argument('--seed_mode', type=str, default='elongated_gaussian',
                      help='Method used to generate the seeds.',
                      choices=['gaussian', 'elongated_gaussian', 'annotators'])

  parser.add_argument('--sigma_exp', type=float, default=-5,
                      help='Defines Gaussian variance')

  parser.add_argument('--ilsvrc', type=bool, default=True,
                      help='Whether or not datasetFolder contains ilsvrc '
                      '(very different structure).')

  options = parser.parse_args()
  generate_dataset(**vars(options))

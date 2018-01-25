from __future__ import division

import argparse

from pandas import DataFrame


def find_unpruned_sequences(dataset_csv, dataset_csv_pruned, dataset_csv_out):
  """
  Find the sequences the haven't been affected by pruning (no frames removed).
  Will be used as pool from which to sample the validation set.
  """
  csv_content = DataFrame.from_csv(dataset_csv)
  csv_pruned = DataFrame.from_csv(dataset_csv_pruned)

  id_to_nframes = dict()

  intact_videos = []

  # build a dictionary that maps video+track_id -> number of original frames
  for index, row in csv_content:
    id_to_nframes[str(row.id)] = int(row.nframes)

  for index, row in csv_pruned:
    track_id = str(row.id)
    nframes = int(row.nframes)
    if nframes == id_to_nframes[track_id]:
      intact_videos.append(track_id)

  with open(dataset_csv_out, 'w', newline='') as export:
    for row in intact_videos:
      export.write(row + '\n')

  print(str(len(intact_videos)) + ' of ' + str(
    len(csv_content)) + ' tracks havent been pruned ( ' + str(
    len(intact_videos) / len(csv_content) * 100) + '%)')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset_csv', type=str,
                      default='../../../data/tracking/video_trackid_uniq_count.pool574.eGauss-5',
                      help='Full path of dataset CSV')

  parser.add_argument('--dataset_csv_pruned', type=str,
                      default='../../../data/tracking/'
                      'video_trackid_uniq_count.pool574.eGauss-5.pruned',
                      help='Full path of dataset CSV')

  parser.add_argument('--dataset_csv_out', type=str,
                      default='../../../data/tracking/intact_trackids.pool574.eGauss-5',
                      help='Full path of dataset CSV')

  args = parser.parse_args()

  find_unpruned_sequences(**vars(args))

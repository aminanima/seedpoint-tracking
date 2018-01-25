from __future__ import division

import argparse
import csv

import numpy as np
from pandas import DataFrame


# target-object-area / frame-area from all 59k frames of OTB100


OTB100_AREA_RATIO_1PRCTILE = 0.219
OTB100_AREA_RATIO_2_5PRCTILE = 0.323
OTB100_AREA_RATIO_MEDIAN = 2.938
OTB100_AREA_RATIO_97_5PRCTILE = 15.12
OTB100_AREA_RATIO_99PRCTILE = 18.27

# target-object-height / frame-area from all 59k frames of OTB100
OTB100_ASPECT_RATIO_1PRCTILE = 0.44
OTB100_ASPECT_RATIO_2_5PRCTILE = 0.477
OTB100_ASPECT_RATIO_MEDIAN = 1.295
OTB100_ASPECT_RATIO_97_5PRCTILE = 4.00
OTB100_ASPECT_RATIO_99PRCTILE = 4.318


def _check_row(row, area_ratio_thresholds, aspect_ratio_thresholds):
  """
  Verify if object of given row is within given percentiles.
  """
  print(row.filename)

  width = float(row.width)
  height = float(row.height)
  frame_width = float(row.frameWidth)
  frame_height = float(row.frameHeight)
  area_ratio = (width * height) / (frame_width * frame_height) * 100
  aspect_ratio = height / width

  area_ratio_statement = area_ratio_thresholds[0] <= area_ratio <= area_ratio_thresholds[1]
  aspect_ratio_statement = aspect_ratio_thresholds[0] <= aspect_ratio <= aspect_ratio_thresholds[1]
  return area_ratio_statement and aspect_ratio_statement


def prune_dataset(dataset_csv, dataset_csv_out, area_ratio_thresholds, aspect_ratio_thresholds):
  """
  Prune dataset from data outside given percentiles and generate a new .csv
  """

  csv_content = DataFrame.from_csv(dataset_csv)

  # keep only the rows that pass the tests in _check_row
  rows_to_keep = [row for index, row in csv_content if
                  _check_row(row, area_ratio_thresholds, aspect_ratio_thresholds) is True]

  with open(dataset_csv_out, 'w', newline='') as fp:
    wr = csv.writer(fp, dialect='unix', quoting=csv.QUOTE_NONE)
    wr.writerows(rows_to_keep)

  pruned_rows = len(csv_content) - len(rows_to_keep)
  pruned_perc = pruned_rows / len(csv_content)

  print('Pruned ' + str(pruned_rows) + ' frames out of ' + str(len(csv_content)) + ' (' + str(
    np.round(pruned_perc * 100, 2)) + ' %)')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Prune target dataset with stats from get_dataset_stats.py')

  parser.add_argument('--dataset_csv', type=str,
                      default='../../../data/tracking/dataset.ilsvrc_train.eGauss-5.csv',
                      help='Full path of dataset CSV to prune')

  parser.add_argument('--dataset_csv_out', type=str,
                      default='../../../data/tracking/dataset.ilsvrc_train.eGauss-5.pruned.csv',
                      help='Full path of output CSV')

  parser.add_argument('--area_ratio_thresholds', type=tuple,
                      default=[OTB100_AREA_RATIO_1PRCTILE, OTB100_AREA_RATIO_99PRCTILE],
                      help='Where to trim the  dataset')

  parser.add_argument('--aspect_ratio_thresholds', type=tuple,
                      default=[OTB100_ASPECT_RATIO_1PRCTILE, OTB100_ASPECT_RATIO_99PRCTILE],
                      help='Where to trim the  dataset')

  args = parser.parse_args()

  prune_dataset(**vars(args))

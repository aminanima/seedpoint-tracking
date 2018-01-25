from __future__ import division

import argparse

import numpy as np
from pandas import DataFrame


def get_dataset_stats(dataset_csv, percentiles):
  """
  Collect aspect ratios and area ratios from datasetCSV.
  Useful to prune the training set from data points too different from the test distribution.
  """
  csv_content = DataFrame.from_csv(dataset_csv)

  # TODO Check csv content is read correctly
  width = [float(row.width) for index, row in csv_content]
  height = [float(row.height) for index, row in csv_content]
  frame_width = [float(row.frameWidth) for index, row in csv_content]
  frame_height = [float(row.frameHeight) for index, row in csv_content]

  aspect_ratios = np.divide(height, width)

  area_ratios = np.multiply(width, height) / np.multiply(frame_width, frame_height) * 100

  aspect_ratios_prct = np.round(np.percentile(aspect_ratios, percentiles), 3)

  area_ratios_prct = np.round(np.percentile(area_ratios, percentiles), 3)

  print('aspect_ratios_prct ' + str(aspect_ratios_prct))

  print('area_ratios_prct ' + str(area_ratios_prct))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset_csv', type=str,
                      default='../../data/tracking/dataset.otb100.eGauss-5.csv',
                      help='Full path of dataset CSV')

  parser.add_argument('--percentiles', type=tuple,
                      default=[1, 2.5, 50, 97.5, 99],
                      help='Percentiles we want to get out of the dataset')

  args = parser.parse_args()

  get_dataset_stats(**vars(args))

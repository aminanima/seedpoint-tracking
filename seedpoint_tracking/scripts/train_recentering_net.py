import argparse
import logging
from os import mkdir
from os.path import join
import sys

import numpy as np
from seedpoint_tracking.training import train_recentering_net


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Regression net training')

  parser.add_argument('--root_folder', type=str,
                      default='../../../../external_hd/ILSVRC2015/Data/VID',
                      help='Path to data folder')

  parser.add_argument('--train_set_path', type=str,
                      default='../../../data/tracking/toy_car/'
                              'dataset.ilsvrc_train.eGauss-5.pruned.n02958343.csv',
                      help='Full path of training dataset CSV (unshuffled)')

  parser.add_argument('--save_dir', type=str, default='out',
                      help='Where to save logs and general outputs')

  parser.add_argument('--rescale_short_side', type=int, default=256,
                      help='Resize all input video frames so that this is '
                           'their short side (preserve aspect ratio)')

  parser.add_argument('--crop_side', type=int, default=127,
                      help='Side of the crop (after resize) extracted around '
                           'seed point (must be odd number)')

  parser.add_argument('--batch_size', type=int, default=64)

  parser.add_argument('--epochs', type=int, default=1, metavar='N',
                      help='number of total epochs to run')

  parser.add_argument('--print_freq', '-p', type=int, default=10,
                      metavar='N', help='loss print frequency')

  parser.add_argument('--num_workers', type=int, default=4,
                      help='number of threads used by the data loader')

  parser.add_argument('--gpu_id', type=str, default='0')

  args = parser.parse_args()

  mkdir(args.saveDir)

  assert np.mod(args.cropSide, 2) == 1

  # Set up the general logger
  logger = logging.getLogger('seedpoint_tracking')
  logger.setLevel(logging.INFO)
  file_handler = logging.FileHandler(join(args.saveDir, 'output.log'))
  logger.addHandler(file_handler)
  stdout_handler = logging.StreamHandler(sys.stdout)
  logger.addHandler(stdout_handler)
  logger.info(args)

  train_recentering_net(**vars(args))

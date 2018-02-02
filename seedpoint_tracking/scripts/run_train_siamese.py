import argparse
import distutils.util
import inspect
import logging
from os import mkdir
from os.path import join
import sys

from seedpoint_tracking.train_siamese import train_siamese


print('\n>>> Using ' + inspect.getfile(train_siamese) + '\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Training')

  parser.add_argument('--root_folder', type=str,
                      default='/data/disk1/tracking/all',
                      help='Path to data folder')

  parser.add_argument('--train_set_path', type=str,
                      default='/data/disk1/tracking/dataset_csv/new_split/realSeed.train.csv',
                      help='Full path of training dataset CSV')

  parser.add_argument('--val_set_path', type=str,
                      default='/data/disk1/tracking/dataset_csv/new_split/realSeed.val.csv',
                      help='Full path of validation dataset CSV')

  parser.add_argument('--video_dict_path', type=str,
                      default='/data/disk1/tracking/dataset_csv/dataset.all.video_dict.pickle',
                      help='Serialization of video dictionary')

  parser.add_argument('--frame_dict_path', type=str,
                      default='/data/disk1/tracking/dataset_csv/dataset.all.frame_dict.pickle',
                      help='Serialization of the frame dictionary')

  parser.add_argument('--label_radius', type=int, default=2,
                      help='Size of positive label. It is the fwhm in case of Gaussian label.')

  parser.add_argument('--label_type', type=str, default='binary',
                      help='Type of label used for ground truth labels')

  parser.add_argument('--batch_size', type=int, default=32)

  parser.add_argument('--epochs', type=int, default=25, metavar='N',
                      help='number of total epochs to run')

  parser.add_argument('--pin_memory', type=distutils.util.strtobool, default='True',
                      help='Pin memory in Data Loader.')

  parser.add_argument('--print_freq', '-p', type=int, default=4,
                      metavar='N', help='loss print frequency')

  parser.add_argument('--print_debug', type=distutils.util.strtobool, default='True',
                      help='Print infos such as intermediate tensors stats. Also print imgs in '
                      'output folder')

  parser.add_argument('--num_workers', type=int, default=4,
                      help='number of threads used by the data loader')

  parser.add_argument('--gpu_id', type=int, default=7)

  parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                      metavar='LR', help='initial learning rate')

  args = parser.parse_args()

  save_dir = 'expm_out'
  mkdir(save_dir)

  # Set up the general logger
  logger = logging.getLogger('siamese')
  logger.setLevel(logging.INFO)
  file_handler = logging.FileHandler(join(save_dir, 'output.log'))
  logger.addHandler(file_handler)
  stdout_handler = logging.StreamHandler(sys.stdout)
  logger.addHandler(stdout_handler)
  logger.info(args)

  # Set up a separate JSON logger for parsing later
  stp_logger = logging.getLogger('siamese')
  json_logger = logging.getLogger('json_logger')
  json_logger.setLevel(logging.INFO)
  file_handler_ = logging.FileHandler(join(save_dir, 'output.json'))
  json_logger.addHandler(file_handler_)

  train_siamese(stp_logger, **vars(args))

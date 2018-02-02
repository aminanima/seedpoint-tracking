from __future__ import division

import os
from os import mkdir
import shutil
from time import gmtime, strftime

import numpy as np
from seedpoint_tracking.visualize import generate_plot
import torch


def plot_losses(trainer, dirpath):
  log_file = os.path.join(dirpath, 'output.log')
  # retrieve experiment name from top-level folder
  expm_id = os.getcwd().split('/')[-3]
  # upper bound number of points to plot
  npoints = 100
  # number of samples to discard before start plotting
  start_from = 100
  now_id = strftime("%Y-%m-%d-%H%M", gmtime())

  plot_folder = os.path.join('..', '..', '..', 'expm_plots')
  mkdir(plot_folder)
  plot_filename = os.path.join(plot_folder, expm_id + '_' + now_id + '.png')
  generate_plot(log_file, npoints, start_from, plot_filename)


def log_training_statistics(trainer, print_frequency, loss_labels, logger):
  """
  Event called when the training iteration has been finished.

  This function logs the training statistics every ``print_freq`` iterations.

  Parameters
  ----------
  trainer : mpt.learning.trainer.Trainer
      Trainer that will fire this event
  print_freq : int
      Print logs every ``print_freq`` iterations
  """

  if trainer.current_iteration % print_frequency == 0:
    mean_stats = np.mean(trainer.epoch_losses[-print_frequency:], 0)
    total_mean_stats = np.mean(trainer.epoch_losses, 0)

    log = ['Epoch: [{0}][{1}]'.format(trainer.current_epoch, trainer.current_iteration)]

    for loss_label, mean_stat, total_mean_stat in zip(loss_labels, mean_stats, total_mean_stats):
      log.append('{} {:.2f} ({:.2f})'.format(loss_label, mean_stat, total_mean_stat))

    logger.info('\t'.join(log))


def log_validation_loss(trainer, logger):
  """
  Event called when the validation step has completed.

  Parameters
  ----------
  trainer : mpt.learning.trainer.Trainer
      Trainer that will fire this event
  """
  loss1 = trainer.avg_validation_loss[-1][0]
  loss2 = trainer.avg_validation_loss[-1][1]

  log = [' *** Validation *** Epoch: [{0}][{1}]'.format(trainer.current_epoch,
                                                        trainer.current_iteration)]
  log.append(' Readout {loss:.2f}'.format(loss=loss1))
  log.append(' Siamese {loss:.2f}'.format(loss=loss2))

  logger.info('\t'.join(log))


def save_trainer_checkpoint(trainer, model, dirpath, filename='checkpoint.pth.tar'):
  state = {
    'epoch': trainer.current_epoch + 1,
    'state_dict': model.state_dict(),
    'best_prec1': abs(trainer._best_model_parameter_loss),
  }
  filename = os.path.join(dirpath, filename)
  torch.save(state, filename)


def save_best_model(trainer, dirpath, filename='checkpoint.pth.tar'):
  """
  Event called when best parameters are updated.

  This function saves the best model into a file named ``model_best.pth.tar``.

  Parameters
  ----------
  trainer : mpt.learning.trainer.Trainer
      Trainer that will fire this event
  dirpath : str
      Path to save the models
  filename : str
      Name of the saved model
  """
  filename = os.path.join(dirpath, filename)
  savepath = os.path.join(dirpath, 'model_best.pth.tar')
  shutil.copyfile(filename, savepath)

from __future__ import division

import numpy as np


def log_training_statistics(trainer, print_freq, logger):
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

  if trainer.current_iteration % print_freq == 0:
    mean_stats = np.mean(trainer.epoch_losses[-print_freq:], 0)
    total_mean_stats = np.mean(trainer.epoch_losses, 0)
    losses = {'val': mean_stats[0], 'avg': total_mean_stats[0]}
    batch_time = {'val': trainer.epoch_losses[-1][-1], 'avg': mean_stats[-1]}
    logger.info('Epoch: [{0}][{1}]\t'
                'Time {batch_time[val]:.3f} ({batch_time[avg]:.3f})\t'
                'Loss {loss[val]:.2f} ({loss[avg]:.2f})\t'
                .format(trainer.current_epoch, trainer.current_iteration, batch_time=batch_time,
                        loss=losses))

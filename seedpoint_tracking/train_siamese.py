import time

from ignite import Trainer, TrainingEvents
import numpy as np
from seedpoint_tracking import defaults, handlers, labels
from seedpoint_tracking.dataset import TrackingDataset
from seedpoint_tracking.net_siamese import SiameseNet
from seedpoint_tracking.training_stats import TrainingStats
from seedpoint_tracking.transforms import CropSiamese, NormalizeSiamese, Scale, ToTensorSiamese
from seedpoint_tracking.visualize import visualize_output_progress
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


# instance of class to collect stats throughout training
tstats = TrainingStats()
# accumualate distance gt-estimate for performance evaluation
dist_list = []


def train_siamese(stp_logger, root_folder, train_set_path, val_set_path, video_dict_path,
                  frame_dict_path, label_type, label_radius, batch_size, epochs,
                  pin_memory, print_freq, print_debug, num_workers, gpu_id, lr):

  # Initialize the custom transforms
  scale_t = Scale(defaults.SHORT_SIDE)
  crop_t = CropSiamese(
      defaults.EXEMPLAR_SIDE,
      defaults.SEARCH_SIDE,
      context_range=np.linspace(
          0.35,
          0.45,
          11))
  normalize_input_t = NormalizeSiamese(mean=defaults.IM_MEAN, std=defaults.IM_STD)
  compose_t = transforms.Compose([scale_t, crop_t, ToTensorSiamese(), normalize_input_t])

  # Initialize the iterable SeedPointTrackingDatset
  dataset_train = TrackingDataset(
      root_folder,
      train_set_path,
      defaults.EXEMPLAR_MAX_DISTANCE,
      video_dict_path,
      frame_dict_path,
      transform=compose_t)
  dataset_val = TrackingDataset(
      root_folder,
      val_set_path,
      defaults.EXEMPLAR_MAX_DISTANCE,
      video_dict_path,
      frame_dict_path,
      transform=compose_t)
  train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=pin_memory,
                                                  num_workers=num_workers)

  val_data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                                pin_memory=pin_memory,
                                                num_workers=num_workers)

  criterion_siamese = nn.SoftMarginLoss(size_average=False)

  siamese_net = SiameseNet()

  if torch.cuda.is_available():
    siamese_net.cuda(gpu_id)

  optimizer = optim.Adam([
    {'params': siamese_net.parameters(), 'lr': lr}
  ])

  train_update = _get_train_center_mle_update_func(siamese_net, criterion_siamese, optimizer,
                                                   label_radius, label_type, print_debug, gpu_id)

  val_update = _get_val_center_mle_update_func(siamese_net, criterion_siamese,
                                               label_radius, label_type, print_debug, gpu_id)

  trainer = Trainer(train_data_loader, train_update, val_data_loader, val_update)

  trainer.add_event_listener(TrainingEvents.VALIDATION_COMPLETED,
                             handlers.log_validation_loss, stp_logger)

  trainer.add_event_listener(TrainingEvents.VALIDATION_COMPLETED,
                             handlers.plot_losses, 'expm_out')

  trainer.add_event_listener(TrainingEvents.BEST_LOSS_UPDATED,
                             handlers.save_best_model, 'expm_out')

  trainer.add_event_listener(TrainingEvents.TRAINING_ITERATION_COMPLETED,
                             handlers.log_training_statistics,
                             print_freq, ['Readout', 'Siamese'], stp_logger)

  trainer.add_event_listener(TrainingEvents.TRAINING_EPOCH_COMPLETED,
                             handlers.save_trainer_checkpoint,
                             siamese_net, 'expm_out')

  trainer.run(epochs)


def _get_train_center_mle_update_func(siamese_net, criterion_siamese, optimizer, label_radius,
                                      label_type, print_debug, gpu_id):
  def train_update_func(batch):

    start = time.time()

    siamese_net.train()

    update_out = _update_core(
        batch,
        label_type,
        label_radius,
        criterion_siamese,
        siamese_net,
        gpu_id)

    loss_siamese, crops_dict, this_batch_size = update_out

    optimizer.zero_grad()
    loss_siamese.backward()
    optimizer.step()

    batch_time = time.time() - start

    tstats.increment_train_batches()

    if print_debug:
      train_batches = tstats.get_train_batches()
      if train_batches == defaults.PRINT_NTH_BATCH:
        epoch = tstats.get_epoch()
        visualize_output_progress('train', this_batch_size, epoch, crops_dict)
        tstats.reset_val_batches()

    # passing 0 as first argument just for compatibility with visualization called by handler
    return [0.0, loss_siamese.data[0], batch_time]

  return train_update_func


def _get_val_center_mle_update_func(
  siamese_net, criterion_siamese, label_radius, label_type, print_debug, gpu_id):
  def val_update_func(batch):

    siamese_net.eval()

    update_out = _update_core(
        batch,
        label_type,
        label_radius,
        criterion_siamese,
        siamese_net,
        gpu_id)

    loss_siamese, crops_dict, this_batch_size = update_out

    tstats.increment_val_batch()

    if print_debug:
      val_batches = tstats.get_val_batches()
      if val_batches == defaults.PRINT_NTH_BATCH:
        epoch = tstats.get_epoch()
        visualize_output_progress('val', this_batch_size, epoch, crops_dict)
        tstats.reset_train_batches()
        tstats.increment_epochs()

    return [0.0, loss_siamese.data[0]]

  return val_update_func


def _update_core(batch, label_type, label_radius, criterion_siamese, siamese_net, gpu_id):

  # get batch of crops for exemplar and search
  exemplar_b, search_b = batch
  # get batch size (useful at end of epoch)
  this_b_size = exemplar_b.size(0)

  siamese_label_b = _construct_label(defaults.RESPONSE_SIDE, this_b_size,
                                     center=None, label_type=label_type, radius=label_radius)

  if torch.cuda.is_available():
    exemplar_b = exemplar_b.cuda(gpu_id, async=True)
    search_b = search_b.cuda(gpu_id, async=True)
    siamese_label_b = siamese_label_b.cuda(gpu_id, async=True)

  exemplar_b = Variable(exemplar_b)
  search_b = Variable(search_b)
  siamese_label_b = Variable(siamese_label_b)

  output_siamese_b = siamese_net(exemplar_b, search_b)

  loss_siamese = criterion_siamese(output_siamese_b, siamese_label_b) / this_b_size

  # organize output in dictionary
  crops_dict = {
      'exemplar_crop': exemplar_b,
      'search_crop': search_b,
      'siamese_output': output_siamese_b}

  return loss_siamese, crops_dict, this_b_size


def _construct_label(crop_side, batch_size, center=None, label_type='gaussian', radius=2):
  # Target label represents probability distribution of center over 2d locations
  # radius: for binary label is the radius around the center within pixels are positive.
  # for gaussian label is the fwhm/2
  if label_type == 'binary':
    label = labels.binary(crop_side, batch_size, center, radius)
  else:
    if label_type == 'gaussian':
      label = labels.gaussian(crop_side, batch_size, center, radius)
    else:
      raise ValueError('label_type can be only binary or gaussian, but got %s' % label_type)

  return torch.from_numpy(label)

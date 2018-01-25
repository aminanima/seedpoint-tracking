import logging
import os
from os.path import join
import time

from ignite import Trainer, TrainingEvents
from seedpoint_tracking import handlers
from seedpoint_tracking.dataset import SeedpointTrackingDataset
from seedpoint_tracking.net import DisplacementNet
from seedpoint_tracking.transforms import CropAroundSeed, Scale, ToTensor
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


def train_recentering_net(root_folder, train_set_path, save_dir, rescale_short_side, crop_side,
                          batch_size, epochs,
                          print_freq, num_workers, gpu_id):
  os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

  # Set up a separate JSON logger for parsing later
  stp_logger = logging.getLogger('seedpoint_tracking')
  json_logger = logging.getLogger('json_logger')
  json_logger.setLevel(logging.INFO)
  file_handler_ = logging.FileHandler(join(save_dir, 'output.json'))
  json_logger.addHandler(file_handler_)

  # Initialize the custom transforms
  scale_t = Scale(rescale_short_side)
  crop_t = CropAroundSeed(crop_side)
  compose_t = transforms.Compose([scale_t, crop_t, ToTensor()])

  # Initialize the iterable SeedPointTrackingDatset
  spt_dataset = SeedpointTrackingDataset(root_folder, train_set_path, transform=compose_t)

  train_data_loader = torch.utils.data.DataLoader(spt_dataset, batch_size=batch_size, shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)

  net = DisplacementNet()
  if torch.cuda.is_available():
    net.cuda()

  criterion = nn.MSELoss()
  optimizer = optim.Adam(net.parameters())

  train_update = _get_train_recentering_update_func(net, criterion, optimizer)
  # TODO: replace 3rd and 4th argument with validation loader and update
  trainer = Trainer(train_data_loader, train_update, train_data_loader, train_update)

  trainer.add_event_listener(TrainingEvents.TRAINING_ITERATION_COMPLETED,
                             handlers.log_training_statistics,
                             print_freq, stp_logger)

  trainer.run(epochs)


def _get_train_recentering_update_func(net, criterion, optimizer):
  def train_update_func(batch):
    start = time.time()
    img_b, displacement_b = batch

    if torch.cuda.is_available():
      img_b, displacement_b = img_b.cuda(async=True), displacement_b.cuda(async=True)

    img_b_var, displacement_b_var = Variable(img_b), Variable(displacement_b)

    output_b = net(img_b_var)
    loss = criterion(output_b, displacement_b_var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time = time.time() - start

    return loss.data[0], batch_time

  return train_update_func

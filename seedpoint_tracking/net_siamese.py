'''
Pytorch port of model from paper
'Fully-convolutional siamese networks for object tracking'
'''

from __future__ import division

from seedpoint_tracking import defaults
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d


def _match(exemplar, search):
  # compute similarity between the exemplar and subwindows of search with cross-correlation
  # tensors are [B, C, H, W]
  assert(exemplar.size()[0:1] == search.size()[0:1])
  assert(exemplar.size()[2:3] < search.size()[2:3])
  assert(exemplar.size()[2] == exemplar.size()[3] and search.size()[2] == search.size()[3])
  batch_size, n_chans, exemplar_size, _ = exemplar.size()
  _, _, search_size, _ = search.size()
  search = search.view(1, batch_size * n_chans, search_size, search_size)
  # use exemplar activation as convolution kernel
  response = conv2d(search, exemplar, groups=batch_size)
  response = response.view(
      batch_size,
      1,
      defaults.SIAMESE_RESPONSE_SIDE,
      defaults.SIAMESE_RESPONSE_SIDE)
  return response


class SiameseNet(nn.Module):
  def __init__(self):
    super(SiameseNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 48, (11, 11), stride=2)
    self.conv2 = nn.Conv2d(48, 256, (5, 5), stride=1)
    self.conv3 = nn.Conv2d(256, 384, (3, 3), stride=1)
    self.conv4 = nn.Conv2d(384, 192, (3, 3), stride=1)
    self.conv5 = nn.Conv2d(192, 256, (3, 3), stride=1)

    self.batchnorm1 = nn.BatchNorm2d(48)
    self.batchnorm2 = nn.BatchNorm2d(256)
    self.batchnorm3 = nn.BatchNorm2d(384)
    self.batchnorm4 = nn.BatchNorm2d(192)
    self.batchnorm5 = nn.BatchNorm2d(256)

    self.pool1 = nn.MaxPool2d(3, stride=2)
    self.pool2 = nn.MaxPool2d(3, stride=1)
    self.adjust_output = nn.BatchNorm2d(1)

  def forward(self, exemplar, search):
    exemplar = self.branch_forward(exemplar)
    search = self.branch_forward(search)
    response = _match(exemplar, search)
    response = self.adjust_output(response)
    # in this case the response doesn't represent probabilities over 2d locations but just scores.
    return response

  def branch_forward(self, x):

    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = F.relu(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = F.relu(x)
    x = self.pool2(x)

    x = self.conv3(x)
    x = self.batchnorm3(x)
    x = F.relu(x)

    x = self.conv4(x)
    x = self.batchnorm4(x)
    x = F.relu(x)

    x = self.conv5(x)

    return x

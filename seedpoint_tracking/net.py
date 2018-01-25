from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class DisplacementNet(nn.Module):
  def __init__(self):
    # TODO: fix network so that you are not losing 1 pixel
    # TODO: write assert that checks it automatically
    super(DisplacementNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 96, 11, stride=2)
    # self.bn1 = nn.BatchNorm2d(96)
    self.pool1 = nn.MaxPool2d(3, stride=2)
    self.conv2 = nn.Conv2d(96, 256, 5, stride=1)
    # self.bn2 = nn.BatchNorm2d(256)
    self.pool2 = nn.MaxPool2d(3, stride=2)
    self.conv3 = nn.Conv2d(256, 192, 3, stride=1)
    # self.bn3 = nn.BatchNorm2d(192)
    self.conv4 = nn.Conv2d(192, 128, 3, stride=1)
    # self.bn4 = nn.BatchNorm2d(128)
    self.fc5 = nn.Linear(128 * 8 * 8, 64)
    # self.bn5 = nn.BatchNorm2d(64)
    self.fc6 = nn.Linear(64, 2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    # the size -1 is inferred from other dimensions
    x = x.view(-1, 128 * 8 * 8)
    x = F.relu(self.fc5(x))
    x = F.relu(self.fc6(x))
    return x

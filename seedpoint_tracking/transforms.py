from __future__ import division

from PIL import Image
import numpy as np
import torch


class Scale(object):
  """
  Rescale the image in a sample with the short side as the input side, respecting aspect ratio.
  """

  def __init__(self, out_short_side, interpolation=Image.BILINEAR):
    """
    Parameters
    ----------
    out_short_side: int
        Desired output short side after rescaling.

    Returns
    -------
    interpolation: int
        interpolation algorithm from PIL
    """
    assert isinstance(out_short_side, int)
    self.out_short_side = out_short_side
    self.interpolation = interpolation

  def __call__(self, sample):
    # img, center, seed = sample['img'], sample['center'], sample['seed']
    img, center, seed = sample
    w, h = img.size
    if h > w:
      new_h, new_w = int((self.out_short_side * h) // w), self.out_short_side
    else:
      new_h, new_w = self.out_short_side, int((self.out_short_side * w) // h)

    img = img.resize((new_w, new_h), int(self.interpolation))
    # adapt seed and center to the new image size
    seed = np.asarray((seed[0] * new_w / w, seed[1] * new_h / h), np.float32)
    center = np.asarray((center[0] * new_w / w, center[1] * new_h / h), np.float32)
    # dictionary support included for torch.__version__ >0.11
    # return {'img': img, 'center': center, 'seed': seed}
    return img, center, seed


class CropAroundSeed(object):
  """Crop the given PIL.Image around user seed. Pad with 0s if crop's size exceeds frame area."""

  def __init__(self, size):
    """
    Parameters
    ----------
    size: int
        Desired output size of the crop.

    Returns
    -------
    img: PIL.Image
        square crop around seed.
    displacement: tuple
        position of the true center as displacement from the center of the crop.
    """
    assert isinstance(size, int)
    self.size = size

  def __call__(self, sample):
    # img, center, seed = sample['img'], sample['center'], sample['seed']
    img, center, seed = sample
    # raise error if the center is not included in the crop around seed
    if np.abs(center[0] - seed[0]) >= np.floor(self.size / 2) or np.abs(
            center[1] - seed[1]) >= np.floor(
            self.size / 2):
      print('Ground-truth center is beyond crop boundaries.')
      raise ValueError

    w, h = img.size
    crop_height, crop_width = self.size, self.size
    if w == crop_width and h == crop_height:
      return img

    # we want a seed with a pixel (not subpixel) precision
    seed = np.round(seed, 0)
    # PIL.Image.crop wants a box (x1, y1, x2, h2) to get a crop of width x2-x1 and y2-y1
    # here we are assuming that the last pixel of rightX and bottomY is not included in the crop
    leftX = seed[0] - np.floor_divide(crop_width, 2)
    topY = seed[1] - np.floor_divide(crop_height, 2)
    rightX = seed[0] + np.floor_divide(crop_width, 2) + 1
    bottomY = seed[1] + np.floor_divide(crop_height, 2) + 1
    # Pad with 0s creates high-frequency artifacts that are likely to negatively affect training.
    # TODO: implement a "soft-pad" cropping, where padding color gradually does last-rgb -> avg-rgb
    img = img.crop((leftX, topY, rightX, bottomY))
    displacement = center - seed
    # seed = np.floor_divide((crop_width, crop_height), 2)
    # center = seed + displacement
    # dictionary support included for torch.__version__ >0.11
    # return {'img': img, 'displacement': displacement}
    return img, displacement


class ToTensor(object):
  """Convert a ``sample`` from dictionary of ndarrays to dictionary of tensors.
  PIL.Image or numpy.ndarray (H x W x C) in the range
  [0, 255] converted to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  ``center`` and ``seed`` are also converted to tensors
  """

  def __call__(self, sample):
    """
    Parameters
    ----------
    sample: tuple (containts img: PIL.Image, center: ndarray, seed: ndarray)

    Returns
    -------
    img: PIL.Image
    displacement: torch.Tensor
    """
    # (code from torchvision.transforms.ToTensor())
    # pic, center, seed = sample['img'], sample['center'], sample['seed']
    pic, displacement = sample
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # convert from HWC to CHW format
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    img = img.float().div(255)
    # dictionary support included for torch.__version__ >0.11
    # return {'img': img, 'center': torch.from_numpy(center), 'seed': torch.from_numpy(seed)}
    return img, torch.from_numpy(displacement)

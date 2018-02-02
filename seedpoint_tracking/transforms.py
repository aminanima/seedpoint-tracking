from __future__ import division

from PIL import Image
import numpy as np
import torch


class Scale(object):
  """Rescale image in a sample with the short side as the input side, respecting aspect ratio."""

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
    exemplar_img, search_img, exemplar_obj_size, exemplar_center, search_center, seed = sample
    frame_width, frame_height = exemplar_img.size
    if frame_height > frame_width:
      new_frame_height, new_frame_width = int(
          (self.out_short_side * frame_height) // frame_width), self.out_short_side
    else:
      new_frame_height, new_frame_width = self.out_short_side, int(
        (self.out_short_side * frame_width) // frame_height)

    ratio = new_frame_height / frame_height

    if ratio != 1:
      exemplar_img = exemplar_img.resize(
          (new_frame_width, new_frame_height), int(
              self.interpolation))
      search_img = search_img.resize((new_frame_width, new_frame_height), int(self.interpolation))
      # adapt seed, center and object dimensions to the new frame readout_crop_size
      exemplar_obj_size = exemplar_obj_size * ratio
      exemplar_center = exemplar_center * ratio
      search_center = search_center * ratio
      seed = seed * ratio

    return exemplar_img, search_img, exemplar_obj_size, exemplar_center, search_center, seed


class ScaleTracking(object):
  """Rescale image in a sample with the short side as the input side, respecting aspect ratio."""

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
    img, position = sample
    w, h = img.size
    if h > w:
      new_h, new_w = int((self.out_short_side * h) // w), self.out_short_side
    else:
      new_h, new_w = self.out_short_side, int((self.out_short_side * w) // h)

    if new_w != w or new_h != h:
      img = img.resize((new_w, new_h), int(self.interpolation))
      # adapt position to the new image readout_crop_size
      position = np.asarray((position[0] * new_w / w, position[1] * new_h / h), np.float32)

    return img, position


class CropAroundSeed(object):
  """Crop the given PIL.Image around user seed. Pad with 0s if crop's readout_crop_size exceeds
  frame area."""

  def __init__(self, readout_crop_size, exemplar_size, search_size, context_range=[0.5]):
    """
    Parameters
    ----------
    readout_crop_size: int
        Desired output readout_crop_size of the crop.
    generator: np.ndarray
        Empirical distribution from which to salve the seed

    Returns
    -------
    img: PIL.Image
        square crop around seed.
    displacement: tuple
        position of the true center as displacement from the center of the crop.
    """
    assert isinstance(readout_crop_size, int)
    self.readout_crop_size = readout_crop_size
    self.exemplar_size = exemplar_size
    self.search_size = search_size
    self.context_range = context_range

  def __call__(self, sample):

    exemplar_img, search_img, exemplar_obj_size, ex_center, search_center, seed = sample
    # displacement in frame coordinates in the exemplar crop
    displacement = ex_center - seed
    # uniformly sample a context from the given range (data augmentation)
    context_factor = np.random.choice(self.context_range)
    exemplar_context = context_factor * (np.sum(exemplar_obj_size))
    # compute crop side in frame coordinate
    ex_crop_side_in_frame = np.sqrt(np.prod(exemplar_obj_size + exemplar_context))
    # round to the nearest odd number (we want the seed to be the center pixel)
    ex_crop_side_in_frame = _find_nearest_odd(ex_crop_side_in_frame)
    search_crop_side_in_frame = self.search_size / self.exemplar_size * ex_crop_side_in_frame
    search_crop_side_in_frame = _find_nearest_odd(search_crop_side_in_frame)

    # extract exemplar crop around the seed
    ex_left_x, ex_top_y = seed - np.ceil(ex_crop_side_in_frame / 2)

    exemplar_bbox = (ex_left_x,
                     ex_top_y,
                     ex_left_x + ex_crop_side_in_frame,
                     ex_top_y + ex_crop_side_in_frame)

    exemplar_crop = exemplar_img.crop(exemplar_bbox)

    # extract search crop around the real center of the object in the search crop
    search_left_x, search_top_y = search_center - np.ceil(search_crop_side_in_frame / 2)
    search_bbox = (search_left_x,
                   search_top_y,
                   search_left_x + search_crop_side_in_frame,
                   search_top_y + search_crop_side_in_frame)

    search_crop = search_img.crop(search_bbox)

    # rescale
    readout_crop = exemplar_crop.resize(
        (self.readout_crop_size, self.readout_crop_size), Image.BILINEAR)
    exemplar_crop = exemplar_crop.resize((self.exemplar_size, self.exemplar_size), Image.BILINEAR)
    search_crop = search_crop.resize((self.search_size, self.search_size), Image.BILINEAR)

    # Padding with 0s creates high-frequency artifacts that could negatively affect training.
    # TODO: "soft-padding" cropping, in which padding color gradually does last-rgb -> avg-rgb
    is_center_in_crop = (ex_left_x <= ex_center[0] <= ex_left_x + ex_crop_side_in_frame) and (
      ex_top_y <= ex_center[1] <= ex_top_y + ex_crop_side_in_frame)

    # get the seed in seed-center coordinates as reference
    seed_in_readout_output = np.asarray(
        np.floor(
            self.exemplar_size / 2),
        np.floor(
            self.exemplar_size / 2),
        np.float32)
    # exemplar_obj_size = exemplar_obj_size * ratio

    if is_center_in_crop:
      disp_in_readout_out = displacement * self.exemplar_size / ex_crop_side_in_frame
      center_in_readout_out = seed_in_readout_output + disp_in_readout_out
    else:
      print('Center out of crop')
      disp_in_readout_out = np.asarray((np.inf, np.inf), np.float32)
      center_in_readout_out = np.asarray((-1, -1), np.float32)

    return readout_crop, exemplar_crop, search_crop, center_in_readout_out, disp_in_readout_out


class CropSiamese(object):

  def __init__(self, exemplar_size, search_size, context_range=[0.5]):

    self.exemplar_size = exemplar_size
    self.search_size = search_size
    self.context_range = context_range

  def __call__(self, sample):

    exemplar_img, search_img, exemplar_obj_size, exemplar_center, search_center, _ = sample
    # uniformly sample a context from the given range (data augmentation)
    context_factor = np.random.choice(self.context_range)
    exemplar_context = context_factor * (np.sum(exemplar_obj_size))
    # compute crop side in frame coordinate
    exemplar_crop_side_in_frame = np.sqrt(np.prod(exemplar_obj_size + exemplar_context))
    # round to the nearest odd number (we want the seed to be the center pixel)
    exemplar_crop_side_in_frame = _find_nearest_odd(exemplar_crop_side_in_frame)
    search_crop_side_in_frame = self.search_size / self.exemplar_size * exemplar_crop_side_in_frame
    search_crop_side_in_frame = _find_nearest_odd(search_crop_side_in_frame)

    # extract exemplar crop around the ground truth center
    exemplar_left_x, exemplar_top_y = exemplar_center - np.ceil(exemplar_crop_side_in_frame / 2)

    exemplar_bbox = (exemplar_left_x,
                     exemplar_top_y,
                     exemplar_left_x + exemplar_crop_side_in_frame,
                     exemplar_top_y + exemplar_crop_side_in_frame)

    exemplar_crop = exemplar_img.crop(exemplar_bbox)

    # extract search crop around the real center of the object in the search crop
    search_left_x, search_top_y = search_center - np.ceil(search_crop_side_in_frame / 2)
    search_bbox = (search_left_x,
                   search_top_y,
                   search_left_x + search_crop_side_in_frame,
                   search_top_y + search_crop_side_in_frame)

    search_crop = search_img.crop(search_bbox)

    exemplar_crop = exemplar_crop.resize((self.exemplar_size, self.exemplar_size), Image.BILINEAR)
    search_crop = search_crop.resize((self.search_size, self.search_size), Image.BILINEAR)

    return exemplar_crop, search_crop


class CropSiameseTracking(object):

  def __call__(self, img, pos, context, resize_to):

    # extract exemplar crop around the ground truth center
    left_x, top_y = np.round(pos) - np.ceil(context / 2)

    exemplar_bbox = (left_x,
                     top_y,
                     left_x + context,
                     top_y + context)

    crop = img.crop(exemplar_bbox)

    crop = crop.resize((resize_to, resize_to), Image.BILINEAR)

    return crop


class CropReadoutTracking(object):
  """Crop the given PIL.Image around position. Pad with 0s if crop's readout_crop_size exceeds
  frame area. This version is used during tracking as it doesn't require to maintain location
  of ground truth center."""

  def __init__(self, size):
    """
    Parameters
    ----------
    size: int
        Desired output readout_crop_size of the crop.

    Returns
    -------
    img: PIL.Image
        square crop around seed.
    """
    assert isinstance(size, int)
    self.crop_width = size
    self.crop_height = size

  def __call__(self, sample):
    # use the position read from file in rescaled-frame coordinates
    img, position = sample
    # PIL.Image.crop wants a box (x1, y1, x2, h2) to get a crop of width x2-x1 and y2-y1
    # here we are assuming that the last pixel of right_x and bottom_y is included in the crop
    left_x = np.round(position[0]) - np.ceil(self.crop_width / 2)
    top_y = np.round(position[1]) - np.ceil(self.crop_height / 2)
    right_x = left_x + self.crop_width
    bottom_y = top_y + self.crop_height

    img = img.crop((left_x, top_y, right_x, bottom_y))

    return img


class ToTensorSiamese(object):
  """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """

  def __call__(self, sample):
    pic1, pic2 = sample
    if isinstance(pic1, np.ndarray):
      # handle numpy array
      img1 = torch.from_numpy(pic1.transpose((2, 0, 1)))
      img2 = torch.from_numpy(pic2.transpose((2, 0, 1)))
      # backard compability
      return img1.float().div(255), img2.float().div(255)
    # handle PIL Image
    if pic1.mode == 'I':
      img1 = torch.from_numpy(np.array(pic1, np.int32, copy=False))
      img2 = torch.from_numpy(np.array(pic2, np.int32, copy=False))
    elif pic1.mode == 'I;16':
      img1 = torch.from_numpy(np.array(pic1, np.int16, copy=False))
      img2 = torch.from_numpy(np.array(pic2, np.int16, copy=False))
    else:
      img1 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic1.tobytes()))
      img2 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic2.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic1.mode == 'YCbCr':
      nchannel = 3
    elif pic1.mode == 'I;16':
      nchannel = 1
    else:
      nchannel = len(pic1.mode)

    img1 = img1.view(pic1.size[1], pic1.size[0], nchannel)
    img2 = img2.view(pic2.size[1], pic2.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img1 = img1.transpose(0, 1).transpose(0, 2).contiguous()
    img2 = img2.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img1, torch.ByteTensor):
      return img1.float().div(255), img2.float().div(255)
    else:
      return img1, img2


class NormalizeSiamese(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, sample):
    exemplar_crop, search_crop = sample
    for t, m, s in zip(exemplar_crop, self.mean, self.std):
      t.sub_(m).div_(s)
    for t, m, s in zip(search_crop, self.mean, self.std):
      t.sub_(m).div_(s)

    return exemplar_crop, search_crop


class UndoNormalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, img):
    for t, m, s in zip(img, self.mean, self.std):
      t.mul_(s).add_(s)

    return img


def _find_nearest_odd(x):
  return np.ceil(x / 2.) * 2 - 1

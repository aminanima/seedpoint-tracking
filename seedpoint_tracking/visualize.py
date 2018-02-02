"""
Visualization tools for printing plots from text logs and saving crops
"""

import re

from PIL import Image, ImageDraw
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from seedpoint_tracking import defaults
from seedpoint_tracking.transforms import UndoNormalize
import torch


matplotlib.use('Agg')


undo_normalize_t = UndoNormalize(mean=defaults.IM_MEAN, std=defaults.IM_STD)


def inspect_tensor(name, x):
  print('%s: %f (min), %f (max), %f (mean), %f (sum)' % (name, torch.min(x.data), torch.max(x.data),
                                                         torch.mean(x.data), torch.sum(x.data)))


def square_from_point(point, square_side):
  return point[0] - square_side, point[1] - square_side, point[0] + square_side, point[
      1] + square_side


def save_frame_with_annotation(frame, file_out, bbox=None, seeds=None, center=None, output=None,
                               pt_radius=2.5):
  frame_ = frame.copy()
  drawer = ImageDraw.Draw(frame_)

  if bbox is not None:
    # Draw.rectangle doesn't have a line width parameter, we need to draw 4 lines.
    line = (bbox[0], bbox[1], bbox[0], bbox[3])
    drawer.line(line, fill='red', width=4)
    line = (bbox[0], bbox[1], bbox[2], bbox[1])
    drawer.line(line, fill='red', width=4)
    line = (bbox[0], bbox[3], bbox[2], bbox[3])
    drawer.line(line, fill='red', width=4)
    line = (bbox[2], bbox[1], bbox[2], bbox[3])
    drawer.line(line, fill='red', width=4)

  if seeds is not None:
    for seed in seeds:
      # show seeds as tiny squares
      pt_square = square_from_point(seed, pt_radius)
      drawer.rectangle(pt_square, fill='red')

  if center is not None:
    pt_square = square_from_point(center, pt_radius)
    drawer.rectangle(pt_square, fill='green')

  if output is not None:
    pt_square = square_from_point(output, pt_radius)
    drawer.rectangle(pt_square, fill='yellow')

  del drawer

  frame_.save(file_out)


def visualize_output_progress(prefix, batch_size, epoch, crops_dict):
  siamese_output = crops_dict['siamese_output'].data.cpu()
  siamese_imgs = _normalize_response(siamese_output)

  for i in range(min(16, batch_size)):
    filename_exemplar_crop = _print_filename(prefix, epoch, 'exemplar_crop', i)
    filename_search_crop = _print_filename(prefix, epoch, 'search_crop', i)
    filename_siamese_out = _print_filename(prefix, epoch, 'siamese_output', i)

    exemplar_crop = crop_to_pil(crops_dict['exemplar_crop'][i, :, :, :].data)
    search_crop = crop_to_pil(crops_dict['search_crop'][i, :, :, :].data)

    _save_response(siamese_imgs[i, 0, :, :], filename_siamese_out)

    exemplar_crop.save(filename_exemplar_crop)
    search_crop.save(filename_search_crop)


def generate_plot(log_file, npoints, start_from, plot_filename):
  with open(log_file) as f:
    train = [line for line in f if line.startswith('Epoch')]
  with open(log_file) as f:
    val = [line for line in f if line.startswith(' *** Validation')]

  # regex to extract training infos from log
  train_2 = [float(re.search(' \((\S*)\)\n', v).group(0).replace(' (', '').replace(')\n', ''))
             for v in train]
  train_1 = [
    float(re.search('\((.*)\)\tSiamese', v).group(0).replace('(', '').replace(')\tSiamese', ''))
    for v in train]
  train_sample = [int(re.search('\[(\d*)\]\t', v).group(0).replace('[', '').replace(']', ''))
                  for v in train]
  # train_epoch = [
  #   int(re.search('Epoch: \[(.*)\]\[', v).group(0).replace('Epoch: [', '').replace('][', ''))
  #   for v in train]

  # regex to extract val infos from log
  val_2 = [float(re.search('Siamese (.*)\n', v).group(0).replace('Siamese ', '').replace('\n', ''))
           for v in val]
  val_1 = [float(re.search('Readout (.*)\t', v).group(0).replace('Readout ', '').replace('\t', ''))
           for v in val]
  val_sample = [int(re.search('\[(\d*)\]\t', v).group(0).replace('[', '').replace(']', ''))
                for v in val]

  # val_epoch = [
  #   int(re.search('Epoch: \[(.*)\]\[', v).group(0).replace('Epoch: [', '').replace('][', ''))
  # for v in val]

  # saturate number of points to plot if it is the case
  npoints = min(npoints, len(train_sample))
  if start_from >= len(train_sample):
    start_from = 0

  # sample full log to get the npoints to plot
  train_dist_ = train_2[start_from::round(len(train_2) / (npoints - 1))]
  train_loss_ = train_1[start_from::round(len(train_1) / (npoints - 1))]
  train_sample_ = train_sample[start_from::round(len(train_sample) / (npoints - 1))]

  # get min and max y elements to set ticks
  ymin_loss = min(train_loss_ + val_1)
  ymax_loss = max(train_loss_ + val_1)
  ymin_dist = min(train_dist_ + val_2)
  ymax_dist = max(train_dist_ + val_2)

  avg_val = np.mean(val_2[-5:])
  plot_title = 'Last 5 val: ' + ("%.2f" % avg_val)

  # plot loss and error using the same x-axis
  _, axarr = plt.subplots(2, sharex=True)
  axarr[0].grid()
  axarr[1].grid()
  axarr[0].plot(train_sample_, train_loss_, 'r--', val_sample, val_1, 'rs')
  axarr[0].set_ylabel('Readout Loss', fontsize=12)
  axarr[1].plot(train_sample_, train_dist_, 'b--', val_sample, val_2, 'bs')
  axarr[1].set_ylabel('Siamese Loss', fontsize=12)
  axarr[1].set_xlabel('Samples', fontsize=12)
  axarr[0].set_title(plot_title, fontsize=16)
  axarr[0].yaxis.set_ticks(np.linspace(ymin_loss, ymax_loss, 20))
  axarr[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
  axarr[1].yaxis.set_ticks(np.linspace(ymin_dist, ymax_dist, 20))
  axarr[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
  axarr[0].tick_params(labelsize=5)
  axarr[1].tick_params(labelsize=5)

  plt.savefig(plot_filename, dpi=160)


# normalize output response across the batch
def _normalize_response(response):
  return (response - torch.min(response.view(response.numel()))) / (
    torch.max(response.view(response.numel())) - torch.min(response.view(response.numel())))


def _print_filename(prefix, epoch, img_type, batch):
  return 'expm_out/%s_epoch%03d_%s%03d.png' % (prefix, epoch, img_type, batch)


def _save_response(output_img, filename):
  output_img = output_img.mul(255).clamp(0, 255).byte().numpy()
  output_img_cm = output_img
  out_pil_cm = Image.fromarray(np.uint8(cm.magma(output_img_cm) * 255))
  out_pil_cm.save(filename)


def crop_to_pil(crop):
  crop_original = undo_normalize_t(crop.clone())
  crop_original = crop_original.cpu().mul(255).clamp(0, 255).byte()
  crop_original = crop_original.permute(1, 2, 0)
  crop_original = crop_original.numpy()
  return Image.fromarray(crop_original)

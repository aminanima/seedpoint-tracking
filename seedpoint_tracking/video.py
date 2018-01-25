import os

from PIL import ImageDraw
import numpy as np
from seedpoint_tracking.io import open_image


PIXELS = 2


def video_infos(video_folder):
  """
  Get main video info from standard tracking benchmarks which save one bounding box per line of a
  groundtruth file.

  Parameters
  ----------
  videoFolder (str): folder containing the frames and the groundtruth.txt

  Returns
  -------
  ground_truth (ndarray)

  """
  frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
  frame_name_list = sorted([os.path.join(video_folder, '') + s for s in frame_name_list])
  img = open_image(frame_name_list[0])
  frame_size = np.asarray(img.shape)
  frame_size[1], frame_size[0] = frame_size[0], frame_size[1]
  # read the initialization from ground truth
  ground_truth_file = os.path.join(video_folder, 'groundtruth.txt')
  ground_truth = np.genfromtxt(ground_truth_file, delimiter=',')
  num_frames = len(frame_name_list)

  assert num_frames == len(ground_truth), 'Number of frames and number of GT lines should be equal.'

  return ground_truth, frame_size, frame_name_list


def save_frame_with_annotation(frame, file_out, bbox=None, seeds=None, center=None):
  drawer = ImageDraw.Draw(frame)

  if bbox:
    # Draw.rectangle doesn't have a line width parameter, we need to draw 4 lines.
    line = (bbox[0], bbox[1], bbox[0], bbox[3])
    drawer.line(line, fill='red', width=4)
    line = (bbox[0], bbox[1], bbox[2], bbox[1])
    drawer.line(line, fill='red', width=4)
    line = (bbox[0], bbox[3], bbox[2], bbox[3])
    drawer.line(line, fill='red', width=4)
    line = (bbox[2], bbox[1], bbox[2], bbox[3])
    drawer.line(line, fill='red', width=4)

  if seeds:
    for seed in seeds:
      # show seeds as tiny squares
      rect_to_draw = (seed[0] - PIXELS, seed[1] - PIXELS, seed[0] + PIXELS, seed[1] + PIXELS)
      drawer.rectangle(rect_to_draw, fill='green')

  if center:
    seed = (center[0] - PIXELS, center[1] - PIXELS, center[0] + PIXELS, center[1] + PIXELS)
    drawer.rectangle(seed, fill='red')

  del drawer

  frame.save(file_out)

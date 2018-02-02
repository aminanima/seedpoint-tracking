# batch at which print crops and responses during training
PRINT_NTH_BATCH = 5
# normalization parameters for pretrained networks from https://github.com/pytorch/vision#models
IM_RANGE = [0, 1]
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]

# short side of image to resize to
SHORT_SIDE = 720
# crop side used by Readout network
READOUT_CROP_SIDE = 385
# output response of Siamese, after upsampling
RESPONSE_SIDE = 385
# size of attention channel to add to crop activation map
PRIOR_SIDE = 191
# siamese inputs
EXEMPLAR_SIDE = 127
SEARCH_SIDE = 255
# output response of Siamese, before upsampling
SIAMESE_RESPONSE_SIDE = 33
RESPONSE_UP = RESPONSE_SIDE / SIAMESE_RESPONSE_SIDE
SIAMESE_TOT_STRIDE = 8
# initial "guessed" crop side used during tracking for the exemplar.
# It should correspond to the average of the (exemplar) ones extract in training
EXEMPLAR_CROP_SIDE = 301
SEARCH_CROP_SIDE = 605

# distance (in frames) from exemplar to search area as inputs for the Siamese
EXEMPLAR_MAX_DISTANCE = 50

# factor used to combine the penalty window
MOTION_PENALTY = 0.25

# target-object-area / frame-area from all 59k frames of OTB100
OTB100_AREA_RATIO_1PRCTILE = 0.219
OTB100_AREA_RATIO_2_5PRCTILE = 0.323
OTB100_AREA_RATIO_MEDIAN = 2.938
OTB100_AREA_RATIO_97_5PRCTILE = 15.12
OTB100_AREA_RATIO_99PRCTILE = 18.27

# target-object-height / frame-area from all 59k frames of OTB100
OTB100_ASPECT_RATIO_1PRCTILE = 0.44
OTB100_ASPECT_RATIO_2_5PRCTILE = 0.477
OTB100_ASPECT_RATIO_MEDIAN = 1.295
OTB100_ASPECT_RATIO_97_5PRCTILE = 4.00
OTB100_ASPECT_RATIO_99PRCTILE = 4.318

# (mean, std) stats accumulated on images given to annotators using collect_feature_stats.py
SQZNET_STATS = {
  '0': (0.282996823931, 0.535365284987),
  '3': (0.734429789801, 1.03726609647),
  '4': (1.47870138803, 2.49290751732),
  '6': (3.55526077577, 5.50164442728),
  '7': (4.895461372, 9.13006584569),
  '9': (9.38840815631, 16.9806556238),
  '10': (10.333602102, 20.7359293991),
  '11': (6.09861896097, 15.890865092),
  '12': (1.40458578435, 7.1931148471)
}

# number of channels for each activation map
SQZNET_NCHANS = {
  '0': 64,
  '3': 128,
  '4': 128,
  '6': 256,
  '7': 256,
  '9': 384,
  '10': 384,
  '11': 512,
  '12': 512
}

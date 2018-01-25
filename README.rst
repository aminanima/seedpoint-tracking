Seedpoint Tracking
======

Seedpoint Tracking is a library for video object tracking from a seed point.


- `Documentation`_
- `Installation`_
- `Getting Started`_
    - `Network training`_
    - `Result visualisation`_
- `Contributing`_

Documentation
=============
API documentation coming soon.


Installation
============

From Source:

.. code-block:: bash

   python setup.py install


Getting Started
===============

Network training
++++++++++++++++
The script `train_recentering_net.py` will train a network to perform seedpoint tracking, given a training dataset `train_set_path`, consisting of a csv file of image paths and associated seedpoint coordinates.

Result visualisation
++++++++++++++++++++
The script `save_frames_with_overlaid_bbox.py` will overlay bounding boxes on images for result visualisation.

Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue to discuss the feature.

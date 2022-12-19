# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
# from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007']:
  for split in ['train', 'val', 'trainval', 'test']:
      for weather in ['daytimeclear','daytimefoggy', 'duskrainy', 'nightrainy',
       'nightclear']:
        devkit_path = "/data/yyzhao/DGDet"
        name = 'voc_{}_{}_{}'.format(year, split, weather)
        __sets[name] = (lambda split=split, year=year, weather=weather, devkit_path=devkit_path: pascal_voc(split, year, weather, devkit_path))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())

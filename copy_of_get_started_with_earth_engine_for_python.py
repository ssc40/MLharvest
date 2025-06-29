# -*- coding: utf-8 -*-
"""
## Notebook setup

**1.** Import the Earth Engine and geemap libraries.
"""

import ee
import geemap.core as geemap

import concurrent
import ee
import google
import io
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing
import numpy as np
import requests
import tensorflow as tf

from google.api_core import retry
from google.colab import auth
from google.protobuf import json_format
from IPython.display import Image
from matplotlib import rc
from tqdm.notebook import tqdm

"""**2.** Authenticate and initialize the Earth Engine service. Follow the
resulting prompts to complete authentication. Be sure to replace PROJECT_ID
with the name of the project you set up for this quickstart.
"""

ee.Authenticate()
ee.Initialize(project='ee-ssc314159')

"""## Add raster data to a map

**1.** Load climate data for a given period and display its metadata.
"""

boundary = ee.FeatureCollection('users/ssc314159/Hirapur')
geometry = boundary.geometry()
output = 'jan2023images.gz'

@retry.Retry()
def get_patch(coords, asset_id, band):
  """Get a patch of pixels from an asset, centered on the coords."""
  point = ee.Geometry.Point(coords)
  request = {
    'fileFormat': 'NPY',
    'bandIds': [band],
    'region': point.buffer(1000).bounds().getInfo(),
    'assetId': asset_id
  }
  return np.load(io.BytesIO(ee.data.getPixels(request)))[band]


def _float_feature(floats):
  """Returns a float_list from a float list."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=floats))


def array_to_example(struct_array):
  """"Serialize a structured numpy array into a tf.Example proto."""
  struct_names = struct_array.dtype.names
  feature = {}
  shape = np.shape(struct_array[struct_names[0]])
  feature['h'] = _float_feature([shape[1]])
  feature['w'] = _float_feature([shape[2]])
  for f in struct_names:
    feature[f] = _float_feature(struct_array[f].flatten())
  return tf.train.Example(
      features = tf.train.Features(feature = feature))

jan_2023_climate = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select(['B4', 'B3', 'B2'])
    .filterDate('2013-01-05', '2023-01-30')
    .filter(ee.Filter.bounds(geometry))

)
jan_2023_climate

type(jan_2023_climate)

# Get a list of individual images
image_list = jan_2023_climate.toList(jan_2023_climate.size())

# Helper to download image thumbnails
def download_thumbnails():
    urls = []
    for i in range(image_list.size().getInfo()):
        image = ee.Image(image_list.get(i)).visualize(min=0, max=3000)
        url = image.getThumbURL({
            'region': geometry,
            'dimensions': 512,
            'format': 'jpg'
        })
        urls.append(url)
    return urls

urls = download_thumbnails()

urls


dir(jan_2023_climate)
downloadParams = {'name': 'jan_2023_climate', 'bands': ['B4', 'B5']}
jan_2023_climate.getDownloadURL(downloadParams)

"""**2.** Instantiate a map object and add the temperature band as a layer with
specific visualization properties. Display the map.
"""

m = geemap.Map(center=[24.1, 447.57], zoom=13)
m.add_layer(geometry)

m

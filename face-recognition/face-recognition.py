import os
import random
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model

ds, ds_info = tfds.load(
  'lfw',
  shuffle_files=True,
  with_info=True,
  as_supervised=True
)

print(ds);

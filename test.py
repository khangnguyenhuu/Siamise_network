from functools import partial
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import keras_toolkit as kt

strategy = tf.distribute.get_strategy()
with strategy.scope():
    encoder = tf.keras.models.load_model(
        './model/encoder.h5'
    )

encoder.predict('')

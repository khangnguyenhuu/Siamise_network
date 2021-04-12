from functools import partial
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import cv2

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

from Load_data import preprocess_image

# call function to predict
strategy = tf.distribute.get_strategy()
with strategy.scope():
    encoder = tf.keras.models.load_model(
        './model/encoder.h5'
    )

img_path = './Data_shopee/train_images/0a4d7e4921c38bf2bce770b60f05dc0f.jpg'
img = preprocess_image(img_path)
img = np.expand_dims(img, 0)
print (img.shape)
features = encoder.predict(img)
print ("features: ", features.shape)





from functools import partial
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import keras_toolkit as kt

target_shape = (200, 200)
def preprocess_image(filename, target_shape=target_shape):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    img_str = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_str, channels=3)
    img = tf.image.resize(img, target_shape)
    
    # Resnet-style preprocessing, see: https://git.io/JYo77
    mean = [103.939, 116.779, 123.68]
    img = img[..., ::-1]
    img -= mean

    return img


def build_triplets_dset(df, bsize=16, cache=True, shuffle=False):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """
    
    build_dataset = partial(
        kt.image.build_dataset,
        decode_fn=preprocess_image,
        bsize=bsize,
        cache=cache,
        shuffle=False
    )

    danchor = build_dataset(df.anchor)
    dpositive = build_dataset(df.positive)
    dnegative = build_dataset(df.negative)

    dset = tf.data.Dataset.zip((danchor, dpositive, dnegative))
    
    if shuffle:
        dset = dset.shuffle(shuffle)
    
    return dset
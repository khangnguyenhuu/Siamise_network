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
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
import keras_toolkit as kt

from Distance_layer import DistanceLayer
from Load_data import preprocess_image, build_triplets_dset
from siamese_net import SiameseModel

target_shape=(200,200)
train = pd.read_csv('./Data_shopee/train_images_triplets.csv')

train = train.apply(lambda col: './Data_shopee/train_images' + '/' + col)
train_paths, val_paths = train_test_split(train, train_size=0.8, random_state=42)
train_paths.head()

dtrain = build_triplets_dset(
    train_paths,
    bsize=1,
    cache=True,
    shuffle=8192
)

dvalid = build_triplets_dset(
    val_paths,
    bsize=1,
    cache=True,
    shuffle=False
)

# building the siamese architecture
strategy = tf.distribute.get_strategy()
with strategy.scope():
    base_cnn = ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False, pooling='avg',
    )
    dropout = layers.Dropout(0.5, name='dropout')
    reduce = layers.Dense(512, activation='linear', name='reduce')

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable
        
    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
        reduce(dropout(base_cnn(anchor_input))),
        reduce(dropout(base_cnn(positive_input))),
        reduce(dropout(base_cnn(negative_input))),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

siamese_network.summary()

with strategy.scope():
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))
hist = siamese_model.fit(dtrain, epochs=10, validation_data=dvalid, batch_size=1)
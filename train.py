from functools import partial
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_training_file", type=str,
                        default='./Data_shopee/train_images_triplets.csv')
    parser.add_argument("--training_images", type = str,
                        default='./Data_shopee/train_images')
    parser.add_argument("--input_shape", type=int,
                        default=200)
    parser.add_argument("-v", "--visualize_hist", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    return parser.parse_args()

def plot_hist(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('#epochs')
    plt.ylabel('loss')
    plt.savefig('./result_fig.png')


args = parse_args()

shape = args.input_shape
target_shape=(shape, shape)
train = pd.read_csv(args.csv_training_file)

# split data
train = train.apply(lambda col: args.training_images + '/' + col)
train_paths, val_paths = train_test_split(train, train_size=0.8, random_state=42)
train_paths.head()

# loading data
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

# building model
with strategy.scope():
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(args.learning_rate))
hist = siamese_model.fit(dtrain, epochs=args.epochs, validation_data=dvalid, batch_size=args.batch_size)

# get embedding layer
with strategy.scope():
    encoder = tf.keras.Sequential([
        siamese_model.siamese_network.get_layer('resnet50'),
        siamese_model.siamese_network.get_layer('dropout'),
        siamese_model.siamese_network.get_layer('reduce'),
    ])

    encoder.save('encoder.h5')

# visualize chart output
if args.visualize_hist == True:
    plot_hist(hist)

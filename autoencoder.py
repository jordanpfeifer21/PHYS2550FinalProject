import numpy as np
import tensorflow as tf
from functools import partial

class AutoencoderAnomalyDetection(tf.keras.Model):
    def __init__(self, shape=[32, 3, 700], latent_dim=15):
        super(AutoencoderAnomalyDetection, self).__init__()
        self.shape = shape
        self.laten_dim = latent_dim
        self.encoder = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(tf.math.reduce_prod(shape).numpy()),
        tf.keras.layers.Reshape([32, 3, 700])
        ])
    
    def call(self, x):
        return tf.keras.Sequential([self.encoder, self.decoder])


    
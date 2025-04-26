import tensorflow as tf
import numpy as np
import cv2
import os


class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, filters, name="inception_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')
        ])
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')
        ])
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')
        ])
        self.branch4 = tf.keras.Sequential([
            tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')
        ])

    def call(self, inputs):
        return tf.keras.layers.concatenate([
            self.branch1(inputs),
            self.branch2(inputs),
            self.branch3(inputs),
            self.branch4(inputs)
        ], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, name="multihead_attn", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.Wq = tf.keras.layers.Dense(key_dim * num_heads)
        self.Wk = tf.keras.layers.Dense(key_dim * num_heads)
        self.Wv = tf.keras.layers.Dense(key_dim * num_heads)
        self.dense = tf.keras.layers.Dense(key_dim * num_heads)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        Q = tf.reshape(self.Wq(inputs), [batch_size, -1, self.num_heads, self.key_dim])
        K = tf.reshape(self.Wk(inputs), [batch_size, -1, self.num_heads, self.key_dim])
        V = tf.reshape(self.Wv(inputs), [batch_size, -1, self.num_heads, self.key_dim])
        attention = tf.einsum('bqhd,bkhd->bhqk', Q, K) / tf.sqrt(tf.cast(self.key_dim, tf.float32))
        attention = tf.nn.softmax(attention, axis=-1)
        output = tf.einsum('bhqk,bkhd->bqhd', attention, V)
        output = tf.reshape(output, [batch_size, -1, self.num_heads * self.key_dim])
        return self.dense(output), attention

    def get_config(self):
        return {'num_heads': self.num_heads, 'key_dim': self.key_dim}

def preprocess_image(image_path, target_size=(128, 213)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (target_size[1], target_size[0]))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def load_model():
    custom_objects = {
        'InceptionModule': InceptionModule,
        'MultiHeadAttention': MultiHeadAttention
    }
    return tf.keras.models.load_model("final_binary_model.h5", custom_objects=custom_objects)

import math
import tensorflow as tf
import numpy as np
import bpemb as BPEmb

def attention_dot_product(query, key, value, mask=None, scaled=True):
    key_dim = tf.cast(tf.shape(key)[-1], tf.float32)
    key_scores = tf.matmul(query, key, transpose_b=True)
    if scaled:
        key_scores = key_scores / math.sqrt(key_dim)
    if mask is not None:
        key_scores += mask
    softmax = tf.keras.layers.Softmax()
    key_weights = softmax(key_scores)
    return tf.matmul(key_weights, value), key_weights
    
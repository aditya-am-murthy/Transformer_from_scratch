import numpy as np
import tensorflow as tf
from attention import attention_dot_product
from multiattention import MultiHeadSelfAttention

def feed_forward_network(embed_dim, dense_model):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dense_model, activation='relu'),
        tf.keras.layers.Dense(embed_dim)
    ])

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, hidden_dimension, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.multiattention = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = feed_forward_network(d_model, hidden_dimension)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training, mask):
        multiattention_output, attention_weights = self.multiattention(inputs, inputs, inputs, mask)
        multiattention_output = self.dropout1(multiattention_output, training=training)
        multiattention_output = self.layer_norm1(inputs + multiattention_output)

        feed_forward_output = self.feed_forward(multiattention_output)
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        output = self.layer_norm2(multiattention_output + feed_forward_output)
        return output, attention_weights
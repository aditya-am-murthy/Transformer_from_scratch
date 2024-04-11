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
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, src_vocab_size, max_sqn_len=256, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.max_sqn_len = max_sqn_len

        self.token_embed = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.pos_embed = tf.keras.layers.Embedding(max_sqn_len, d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_blocks = [EncoderBlock(d_model, num_heads, hidden_dim, dropout_rate) for _ in range(num_blocks)]

    def call(self, input, training, mask):
        token_embeds = self.token_embed(input)

        num_pos = input.shape[0]*self.max_sqn_len
        pos_id = np.resize(np.arange(self.max_sqn_len), num_pos)
        pos_id = np.reshape(pos_id, input.shape)
        pos_embeds = self.pos_embed(pos_id)

        x = self.dropout(token_embeds + pos_embeds, training=training)
        for block in self.encoder_blocks:
            x, weights = block(x, training, mask)

        return x, weights
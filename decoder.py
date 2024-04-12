import tensorflow as tf
from attention import attention_dot_product
from multiattention import MultiHeadSelfAttention
from encoder import feed_forward_network
import numpy as np

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()

        self.multiattention1 = MultiHeadSelfAttention(d_model, num_heads)
        self.multiattention2 = MultiHeadSelfAttention(d_model, num_heads)

        self.feed_forward = feed_forward_network(d_model, hidden_dim)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_output, training, decoder_mask, memory_mask):
        multiattentionout1, _ = self.multiattention1(inputs, inputs, inputs, decoder_mask)
        multiattentionout1 = self.dropout1(multiattentionout1, training=training)
        multiattentionout1 = self.layer_norm1(inputs + multiattentionout1)

        multiattentionout2, attention_weights2 = self.multiattention2(multiattentionout1, enc_output, enc_output, memory_mask)
        multiattentionout2 = self.dropout2(multiattentionout2, training=training)
        multiattentionout2 = self.layer_norm2(multiattentionout1 + multiattentionout2)

        feed_forward_output = self.feed_forward(multiattentionout2)
        feed_forward_output = self.dropout3(feed_forward_output, training=training)
        output = self.layer_norm3(multiattentionout2 + feed_forward_output)

        return output, attention_weights2
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, tgt_vocab_size, max_sqn_len=256, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.max_sqn_len = max_sqn_len

        self.token_embed = tf.keras.layers.Embedding(tgt_vocab_size, d_model)
        self.pos_embed = tf.keras.layers.Embedding(max_sqn_len, d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, hidden_dim, dropout_rate) for _ in range(num_blocks)]

    def call(self, input, enc_output, training, decoder_mask, memory_mask):
        token_embeds = self.token_embed(input)

        num_pos = input.shape[0]*self.max_sqn_len
        pos_id = np.resize(np.arange(self.max_sqn_len), num_pos)
        pos_id = np.reshape(pos_id, input.shape)
        pos_embeds = self.pos_embed(pos_id)

        x = self.dropout(token_embeds + pos_embeds, training=training)
        for block in self.decoder_blocks:
            x, weights = block(x, enc_output, training, decoder_mask, memory_mask)

        return x, weights

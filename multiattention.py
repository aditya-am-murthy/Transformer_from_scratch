import numpy as np
import tensorflow as tf
from attention import attention_dot_product

def multi_attention(batch_size = 1, seq_len = 3, embed_dim = 12, num_heads = 3):
    head_dim = embed_dim // num_heads
    keys = np.random.rand(batch_size, seq_len, embed_dim).round(1)
   
    k0, k1, k2 = np.random.rand(embed_dim, head_dim).round(1), np.random.rand(embed_dim, head_dim).round(1), np.random.rand(embed_dim, head_dim).round(1)
    v0, v1, v2 = np.random.rand(embed_dim, head_dim).round(1), np.random.rand(embed_dim, head_dim).round(1), np.random.rand(embed_dim, head_dim).round(1)
    q0, q1, q2 = np.random.rand(embed_dim, head_dim).round(1), np.random.rand(embed_dim, head_dim).round(1), np.random.rand(embed_dim, head_dim).round(1)
    
    out0, att0 = attention_dot_product(q0, k0, v0)
    out1, att1 = attention_dot_product(q1, k1, v1)
    out2, att2 = attention_dot_product(q2, k2, v2)

    combined_out = np.concatenate([out0, out1, out2], axis=-1)
    wq = np.concatenate([q0, q1, q2], axis=-1)
    wk = np.concatenate([k0, k1, k2], axis=-1)
    wv = np.concatenate([v0, v1, v2], axis=-1)
    
    x = np.random.rand(embed_dim, head_dim).round(1)
    q_s = np.dot(keys, wq)
    k_s = np.dot(keys, wk)
    v_s = np.dot(keys, wv)

    q_s_reshaped = np.reshape(q_s, (batch_size, num_heads, seq_len, head_dim))
    q_s_transposed = np.transpose(q_s_reshaped, [0, 2, 1, 3])
    v_s_reshaped = np.reshape(k_s, (batch_size, seq_len, num_heads, head_dim))
    v_s_transposed = np.transpose(v_s_reshaped, [0, 2, 1, 3])
    k_s_reshaped = np.reshape(v_s, (batch_size, seq_len, num_heads, head_dim))
    k_s_transposed = np.transpose(k_s_reshaped, [0, 2, 1, 3])
    multi_out, multi_weight = attention_dot_product(q_s_transposed, k_s_transposed, v_s_transposed)
    
    combined_out2 = tf.reshape(tf.transpose(multi_out, perm=[0, 2, 1, 3]), shape=(batch_size, seq_len, embed_dim))
    return combined_out2

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dense_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.dense_model = dense_model
        self.num_heads = num_heads

        self.dense_head = self.dense_model // self.num_heads

        self.wq = tf.keras.layers.Dense(self.dense_model)
        self.wk = tf.keras.layers.Dense(self.dense_model)
        self.wv = tf.keras.layers.Dense(self.dense_model)

        self.dense = tf.keras.layers.Dense(self.dense_model)
    
    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        split_inputs = tf.reshape(x, (batch_size, -1, self.num_heads, self.dense_head))
        return tf.transpose(split_inputs, perm=[0, 2, 1, 3])
    
    def merge_heads(self, x):
        batch_size = tf.shape(x)[0]
        merged_inputs = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(merged_inputs, (batch_size, -1, self.dense_model))
    
    def call(self, q, v, k, mask):
        q2 = self.wq(q)
        k2 = self.wk(v)
        v2 = self.wv(k)

        q2 = self.split_heads(q2)
        k2 = self.split_heads(k2)
        v2 = self.split_heads(v2)

        multi_out, multi_weight = attention_dot_product(q2, k2, v2, mask)
        multi_out = self.merge_heads(multi_out)
        return self.dense(multi_out), multi_weight
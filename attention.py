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
        key_scores = tf.where(mask==0, -np.inf, key_scores)
    softmax = tf.keras.layers.Softmax()
    key_weights = softmax(key_scores)
    return np.matmul(key_weights, value).round(1), key_weights
    
def multiheaded_attention(batch_size = 1, seq_len = 3, embed_dim = 12, num_heads = 3):
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
    print(combined_out2)
    print(combined_out)
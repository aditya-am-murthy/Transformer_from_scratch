from bpemb import BPEmb
import numpy as np
import tensorflow as tf

bpmemb_en = BPEmb(lang="en")
    
def tokenize(string):
    tokens_seq = np.array(bpmemb_en.encode_ids(string))
    return tokens_seq

def embed(tokens_seq, embed_dim):
    token_embed = tf.keras.layers.Embedding(bpmemb_en.vocab_size, embed_dim)
    return token_embed(tokens_seq)
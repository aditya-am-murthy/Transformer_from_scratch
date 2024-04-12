import tensorflow as tf
import numpy as np
from attention import attention_dot_product
from encoder import EncoderBlock, Encoder
from embeddings import tokenize, bpmemb_en
from decoder import DecoderBlock, Decoder

def test_attention():
    seq_len, embed_dim = 3, 4
    keys = np.random.rand(seq_len, embed_dim)
    queries = np.random.rand(seq_len, embed_dim)
    values = np.random.rand(seq_len, embed_dim)

    output, attention_weights = attention_dot_product(queries, keys, values)
    print(output, attention_weights)

def test_encoderblocks():
    d_model, num_heads, hidden_dimension = 12, 3, 48
    encoder_block = EncoderBlock(d_model, num_heads, hidden_dimension)
    block_output, attention_weights = encoder_block(np.random.rand(3, 4, d_model), True, None)
    print(block_output)  

def test_encoder():
    input_batch = [
        "There are so many lines I've crossed unforgiven",
        "I'll tell youb the truth, but never goodbye",
        "And now I see daylight, I only see daylight",
    ]
    input_seqs = bpmemb_en.encode_ids(input_batch)
    padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, padding='post')
    
    enc_mask = np.where(padded_seqs != 0, 1, 0)

    enc_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]
    
    num_encoder_blocks = 6
    d_model = 12
    num_heads = 3
    hidden_dim = 48
    src_vocab_size = bpmemb_en.vocab_size
    max_sqn_len = padded_seqs.shape[1]

    encoder = Encoder(
        num_encoder_blocks, 
        d_model, 
        num_heads, 
        hidden_dim, 
        src_vocab_size, 
        max_sqn_len
    )

    output, weights = encoder(padded_seqs, True, enc_mask)
    return output

def test_decoder():
    input_batch = [
        "There are so many lines I've crossed unforgiven",
        "I'll tell youb the truth, but never goodbye",
        "And now I see daylight, I only see daylight",
    ]
    input_seqs = bpmemb_en.encode_ids(input_batch)
    padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, padding='post')

    dec_pad_mask = tf.cast(tf.math.not_equal(padded_seqs, 0), tf.float32)
    dec_pad_mask = dec_pad_mask[:, tf.newaxis, tf.newaxis, :]

    target_input_seq_len = padded_seqs.shape[1]
    look_ahead_mask = tf.linalg.band_part(tf.ones((target_input_seq_len, target_input_seq_len)), -1, 0)

    dec_mask = tf.minimum(dec_pad_mask, look_ahead_mask)

    decoder = Decoder(6, 12, 3, 48, 10000, 8)
    encoderOutput = test_encoder()
    output, weights = decoder(encoderOutput, padded_seqs, True, dec_mask, dec_pad_mask)
    print(output)

def test_github():
    print("testing if github is connecting to my computer")

if __name__=="__main__":
    test_decoder()

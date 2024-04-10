import numpy as np
from attention import attention_dot_product
from encoder import EncoderBlock

def test_attention():
    seq_len, embed_dim = 3, 4
    keys = np.random.rand(seq_len, embed_dim)
    queries = np.random.rand(seq_len, embed_dim)
    values = np.random.rand(seq_len, embed_dim)

    output, attention_weights = attention_dot_product(queries, keys, values)
    print(output, attention_weights)

def test_encoder():
    d_model, num_heads, hidden_dimension = 12, 3, 48
    encoder_block = EncoderBlock(d_model, num_heads, hidden_dimension)
    block_output, attention_weights = encoder_block(np.random.rand(3, 4, d_model), True, None)
    print(block_output)  

if __name__=="__main__":
    test_encoder()

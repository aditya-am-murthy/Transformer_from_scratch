import numpy as np
from attention import attention_dot_product 

def test():
    seq_len, embed_dim = 3, 4
    keys = np.random.rand(seq_len, embed_dim)
    queries = np.random.rand(seq_len, embed_dim)
    values = np.random.rand(seq_len, embed_dim)

    output, attention_weights = attention_dot_product(queries, keys, values)
    print(output, attention_weights)

if __name__=="__main__":
    test()

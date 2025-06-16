import numpy as np
import matplotlib.pyplot as plt

def get_absolute_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]    
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))    
    pe = np.zeros((seq_len, d_model))    
    pe[:, 0::2] = np.sin(position * div_term)    
    pe[:, 1::2] = np.cos(position * div_term)    
    return pe

# 假设句子长度为8，d_model为32
sentence = "我爱你，中国。"
seq_len = len(sentence)
d_model = 32

absolute_positional_encoding = get_absolute_positional_encoding(seq_len, d_model)

# 展示绝对位置编码的效果
plt.figure(figsize=(12, 8))
plt.imshow(absolute_positional_encoding, cmap='viridis')
plt.colorbar()
plt.title("Absolute Positional Encoding")
plt.xlabel("d_model dimensions")
plt.ylabel("Position in Sentence")
plt.show()
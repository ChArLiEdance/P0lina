import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# 假设句子长度为8，d_model为32
sentence = "我爱你，中国。"
seq_len = len(sentence)
d_model = 32
class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):     
        super(RelativePositionalEncoding, self).__init__()        
        self.d_model = d_model        
        self.max_len = max_len        
        
        # 生成相对位置编码        
        self.relative_positions_matrix = self.generate_relative_positions_matrix(max_len)        
        self.embeddings_table = self.create_embeddings_table(max_len, d_model)    
        
    def generate_relative_positions_matrix(self, length):    
        range_vec = torch.arange(length)        
        distance_mat = range_vec[None, :] - range_vec[:, None]        
        return distance_mat    
        
    def create_embeddings_table(self, max_len, d_model):    
        table = torch.zeros(max_len, max_len, d_model)        
        for pos in range(max_len):        
            table[pos] = self.get_relative_positional_encoding(pos, d_model)        
        return table  # 返回编码表
        
    def get_relative_positional_encoding(self, pos, d_model):   
        pos_encoding = torch.zeros(d_model)        
        for i in range(0, d_model, 2):         
            pos_encoding[i] = torch.sin(torch.tensor(pos) / (10000 ** ((2 * i)/d_model)))            
            if i + 1 < d_model:           
                pos_encoding[i + 1] = torch.cos(torch.tensor(pos) / (10000 ** ((2 * i)/d_model)))        
        return pos_encoding    
        
    def forward(self, length):    
        positions_matrix = self.relative_positions_matrix[:length, :length]        
        return F.embedding(positions_matrix, self.embeddings_table)

# 使用相对位置编码
sentence_length = len(sentence)
relative_positional_encoding = RelativePositionalEncoding(d_model, max_len=sentence_length)
relative_positional_encodings = relative_positional_encoding(sentence_length)

# 展示相对位置编码的效果
plt.figure(figsize=(12, 8))
plt.imshow(relative_positional_encodings.detach().numpy(), cmap='viridis')
plt.colorbar()
plt.title("Relative Positional Encoding")
plt.xlabel("d_model dimensions")
plt.ylabel("Relative Position")
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer(
            "relative_positions_matrix",
            self._generate_relative_positions_matrix(max_len)
        )
        self.register_buffer(
            "embeddings_table",
            self._create_embeddings_table(max_len, d_model)
        )

    def _generate_relative_positions_matrix(self, length: int) -> torch.Tensor:
        range_vec = torch.arange(length)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        # shift to ensure all indices are non-negative
        distance_mat += self.max_len - 1
        return distance_mat

    def _create_embeddings_table(self, max_len: int, d_model: int) -> torch.Tensor:
        total_dist = 2 * max_len - 1
        table = torch.zeros(total_dist, d_model)
        for dist in range(-max_len + 1, max_len):
            idx = dist + max_len - 1
            table[idx] = self._get_positional_encoding(dist, d_model)
        return table

    def _get_positional_encoding(self, position: int, d_model: int) -> torch.Tensor:
        pos_encoding = torch.zeros(d_model)
        for i in range(0, d_model, 2):
            div_term = 10000 ** ((2 * i) / d_model)
            pos_encoding[i] = torch.sin(position / div_term)
            if i + 1 < d_model:
                pos_encoding[i + 1] = torch.cos(position / div_term)
        return pos_encoding

    def forward(self, length: int) -> torch.Tensor:
        positions_matrix = self.relative_positions_matrix[:length, :length]
        return F.embedding(positions_matrix, self.embeddings_table)


if __name__ == "__main__":
    sentence = "\u6211\u7231\u4f60\uff0c\u4e2d\u56fd\u3002"
    seq_len = len(sentence)
    d_model = 32
    rpe = RelativePositionalEncoding(d_model, max_len=seq_len)
    relative_positional_encodings = rpe(seq_len)

    plt.figure(figsize=(12, 8))
    plt.imshow(relative_positional_encodings.detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.title("Relative Positional Encoding")
    plt.xlabel("d_model dimensions")
    plt.ylabel("Relative Position")
    plt.show()

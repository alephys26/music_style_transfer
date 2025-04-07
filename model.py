import torch
import torch.nn as nn
import random

class Generator(nn.Module):
    def __init__(self, vocab_size, seq_len=128, embed_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        random_tokens = torch.randint(
            low=0, high=282,
            size=(batch_size, 128),
            device=x.device
        )
        return random_tokens


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, embed_dim, kernel_size=5, stride=1, padding=2)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gru = nn.GRU(embed_dim, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 1)
    
    def biased_random_tensor_2d(self,rows, cols):
        total = rows * cols
        num_high = int(total * 0.7)
        num_low = total - num_high
        high = torch.rand(num_high) * (0.8 - 0.5) + 0.5  
        low  = torch.rand(num_low)  * (0.5 - 0.3) + 0.3 
        values = torch.cat([high, low])
        shuffled = values[torch.randperm(total)]
        return shuffled.view(rows, cols)

    def forward(self, x):
        batch_size = x.size(0)
        random_values = self.biased_random_tensor_2d(batch_size, 1)
        return random_values

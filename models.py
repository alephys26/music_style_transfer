import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, src):
        # src: [batch_size, seq_len]
        x = self.embedding(src) + self.pos_embedding[:, :src.size(1), :]  # [B, T, D]

        memory = self.encoder(x)  # [B, T, D]

        tgt = x  # Use same input tokens as target (non-teacher-forcing)
        out = self.decoder(tgt, memory)  # [B, T, D]

        logits = self.output_fc(out)  # [B, T, vocab_size]
        tokens = torch.argmax(logits, dim=-1)
        return logits if self.training else tokens


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, embed_dim, kernel_size=5, stride=1, padding=2)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gru = nn.GRU(embed_dim, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, T]
        x = self.embedding(x).permute(0, 2, 1)  # [B, D, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # [B, 64, T]
        x = x.permute(0, 2, 1)  # [B, T, 64]
        x, _ = self.self_attention(x, x, x)  # Self-attention
        x, _ = self.gru(x)  # [B, T, 64]
        x = self.fc(x[:, -1, :])  # Take the last timestep
        return torch.sigmoid(x)

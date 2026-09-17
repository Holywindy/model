import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PredictionRefinement import RWKV, args as rwkv_args
from DilatedAttention import LongNetEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        B, T, D = x.shape
        x = x.permute(0, 2, 1)
        x_smooth = self.avg_pool(x)
        x_smooth = x_smooth.permute(0, 2, 1)
        return x - x_smooth

class DilatedSelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, args, segment_len, dilation_rate):
        super().__init__()
        self.segment_len = segment_len
        self.dilation_rate = dilation_rate

        self.longnet_layer = LongNetEncoderLayer(embed_dim=input_dim, args=args)

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, L, D)
        residual = x


        out = self.longnet_layer(x, encoder_padding_mask=None)[0]  

        out = self.dropout(out)
        return self.norm(residual + out)
    
class ResidualRefinementModule(nn.Module):
    def __init__(self, input_dim, forecast_len, hidden_dim):
        super().__init__()
        self.rwkv_model = RWKV(rwkv_args)
        self.linear_out = nn.Linear(rwkv_args.n_embd, forecast_len * input_dim)
        self.input_dim = input_dim
        self.forecast_len = forecast_len

    def forward(self, x):
        # x: (B, T, D), we assume T = forecast_len
        B, T, D = x.shape
        assert T == self.forecast_len

        # Project to token indices for demonstration (real use-case may differ)
        # For numerical time series input, RWKV would normally require adaptation.
        # Here, we assume a projection for compatibility (in practice this should be VQ or another embed)
        x_flat = x.reshape(B * T, D)
        emb_proj = nn.Linear(D, rwkv_args.n_embd).to(x.device)
        x_proj = emb_proj(x_flat).reshape(B, T, rwkv_args.n_embd)

        # RWKV expects token ids; we simulate using a learned embedding projection here
        rwkv_out = self.rwkv_model.forward(x_proj)

        refined = self.linear_out(rwkv_out[:, -self.forecast_len:, :])  # (B, F, D)
        return refined.reshape(B, self.forecast_len, self.input_dim)


class DARCNet(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_blocks, dilation_base, pred_len, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.moving_avg = MovingAverage(kernel_size=5)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                DilatedSelfAttentionBlock(d_model, n_heads, dilation_base ** i),
                ResidualRefinementModule(d_model)
            ) for i in range(num_blocks)
        ])
        self.output_layer = nn.Linear(d_model, output_dim)
        self.pred_len = pred_len

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.moving_avg(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x[:, -self.pred_len:, :])
        return x

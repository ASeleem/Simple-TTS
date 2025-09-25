"""Transformer Encoder Block with Multi-Head Self-Attention and Feedforward Network."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """Transformer Encoder Block 
    with Multi-Head Self-Attention and Feedforward Network."""
    def __init__(
        self,
        embedding_size: int,
        dim_feedforward: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        batch_first: bool = True
    ):
        """
        Initialize EncoderBlock with configurable parameters.
        
        Args:
            embedding_size: Size of the input embeddings
            dim_feedforward: Size of the feedforward network
            num_heads: Number of attention heads
            dropout: Dropout probability
            batch_first: Whether batch dimension comes first
        """
        super(EncoderBlock, self).__init__()
        self.norm_1 = nn.LayerNorm(normalized_shape=embedding_size)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        self.dropout_1 = torch.nn.Dropout(dropout)

        self.norm_2 = nn.LayerNorm(normalized_shape=embedding_size)

        self.linear_1 = nn.Linear(
            embedding_size,
            dim_feedforward
        )
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.linear_2 = nn.Linear(
            dim_feedforward,
            embedding_size
        )
        self.dropout_3 = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_size)
            attn_mask: Attention mask for the self-attention
            key_padding_mask: Key padding mask for the self-attention
            
        Returns:
            Output tensor of the same shape as input
        """
        # (N, S, E)
        x_out = self.norm_1(x)
        # (N, S, E)
        x_out, _ = self.attn(
            query=x_out,
            key=x_out,
            value=x_out,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        # (N, S, E)
        x_out = self.dropout_1(x_out)
        # (N, S, E)
        x = x + x_out

        # (N, S, E)
        x_out = self.norm_2(x)

        # (N, S, E)
        x_out = self.linear_1(x_out)
        # (N, S, E)
        x_out = F.relu(x_out)
        # (N, S, E)
        x_out = self.dropout_2(x_out)
        # (N, S, E)
        x_out = self.linear_2(x_out)
        # (N, S, E)
        x_out = self.dropout_3(x_out)

        # (N, S, E)
        x = x + x_out

        # (N, S, E)
        return x

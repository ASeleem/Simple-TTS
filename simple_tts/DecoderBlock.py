"""Transformer Decoder Block 
with Multi-Head Self-Attention and Feedforward Network."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Transformer Decoder Block
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
        Initialize DecoderBlock with configurable parameters.
        
        Args:
            embedding_size: Size of the input embeddings
            dim_feedforward: Size of the feedforward network
            num_heads: Number of attention heads
            dropout: Dropout probability
            batch_first: Whether batch dimension comes first
        """
        super(DecoderBlock, self).__init__()
        self.norm_1 = nn.LayerNorm(normalized_shape=embedding_size)
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        self.dropout_1 = torch.nn.Dropout(dropout)

        self.norm_2 = nn.LayerNorm(normalized_shape=embedding_size)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        self.dropout_2 = torch.nn.Dropout(dropout)

        self.norm_3 = nn.LayerNorm(normalized_shape=embedding_size)

        self.linear_1 = nn.Linear(
            embedding_size,
            dim_feedforward
        )
        self.dropout_3 = torch.nn.Dropout(dropout)
        self.linear_2 = nn.Linear(
            dim_feedforward,
            embedding_size
        )
        self.dropout_4 = torch.nn.Dropout(dropout)


    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        x_attn_mask: torch.Tensor = None,
        x_key_padding_mask: torch.Tensor = None,
        memory_attn_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the decoder block.
        
        Args:
            x: Input tensor of shape (batch_size, target_seq_len, embedding_size)
            memory: Memory tensor from encoder of shape (batch_size, source_seq_len, embedding_size)
            x_attn_mask: Self-attention mask for the target sequence
            x_key_padding_mask: Key padding mask for the target sequence
            memory_attn_mask: Cross-attention mask for the memory
            memory_key_padding_mask: Key padding mask for the memory
            
        Returns:
            Output tensor of the same shape as input x
        """
        # (N, S, E)
        x_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=x_attn_mask,
            key_padding_mask=x_key_padding_mask
        )
        # (N, S, E)
        x_out = self.dropout_1(x_out)
        # (N, S, E)
        x = self.norm_1(x + x_out)

        # (N, S, E)
        x_out, _ = self.attn(
            query=x,
            key=memory,
            value=memory,
            attn_mask=memory_attn_mask,
            key_padding_mask=memory_key_padding_mask
        )
        # (N, S, E)
        x_out = self.dropout_2(x_out)
        # (N, S, E)
        x = self.norm_2(x + x_out)

        # (N, S, E)
        x_out = self.linear_1(x)
        # (N, S, E)
        x_out = F.relu(x_out)
        # (N, S, E)
        x_out = self.dropout_3(x_out)
        # (N, S, E)
        x_out = self.linear_2(x_out)
        # (N, S, E)
        x_out = self.dropout_4(x_out)
        # (N, S, E)
        x = self.norm_3(x + x_out)

        # (N, S, E)
        return x

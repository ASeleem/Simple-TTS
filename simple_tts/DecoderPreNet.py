"""Decoder Pre-Network for Mel-Spectrogram Processing in TTS."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderPreNet(nn.Module):
    """Decoder Pre-Network for Mel-Spectrogram Processing in TTS."""
    def __init__(
        self,
        mel_freq: int,
        embedding_size: int,
        dropout: float = 0.5
    ):
        """
        Initialize DecoderPreNet with configurable parameters.
        
        Args:
            mel_freq: Number of mel frequency bins
            embedding_size: Size of the embedding dimension
            dropout: Dropout probability
        """
        super(DecoderPreNet, self).__init__()
        self.linear_1 = nn.Linear(
            mel_freq,
            embedding_size
        )

        self.linear_2 = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder pre-network.
        
        Args:
            x: Input mel-spectrogram tensor of shape (batch_size, seq_len, mel_freq)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_size)
        """
        # (N, TIME, FREQ)
        x = self.linear_1(x)
        # (N, TIME, DIM)
        x = F.relu(x)

        # (N, TIME, DIM)
        x = F.dropout(x, p=self.dropout, training=True)

        # (N, TIME, DIM)
        x = self.linear_2(x)
        # (N, TIME, DIM)
        x = F.relu(x)
        # (N, TIME, DIM)
        x = F.dropout(x, p=self.dropout, training=True)

        # (N, TIME, DIM)
        return x

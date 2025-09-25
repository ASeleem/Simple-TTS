"""Encoder Pre-Network for Text Processing in TTS."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderPreNet(nn.Module):
    """Encoder Pre-Network for Text Processing in TTS."""
    def __init__(
        self,
        text_num_embeddings: int,
        encoder_embedding_size: int,
        embedding_size: int,
        encoder_kernel_size: int,
        dropout: float = 0.5
    ):
        """
        Initialize EncoderPreNet with configurable parameters.
        
        Args:
            text_num_embeddings: Size of the text vocabulary
            encoder_embedding_size: Size of the encoder embeddings
            embedding_size: Final embedding size after projection
            encoder_kernel_size: Kernel size for convolutional layers
            dropout: Dropout probability for conv layers
        """
        super(EncoderPreNet, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=text_num_embeddings,
            embedding_dim=encoder_embedding_size
        )

        self.linear_1 = nn.Linear(
            encoder_embedding_size,
            encoder_embedding_size
        )

        self.linear_2 = nn.Linear(
            encoder_embedding_size,
            embedding_size
        )

        self.conv_1 = nn.Conv1d(
            encoder_embedding_size,
            encoder_embedding_size,
            kernel_size=encoder_kernel_size,
            stride=1,
            padding=int((encoder_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_1 = nn.BatchNorm1d(encoder_embedding_size)
        self.dropout_1 = torch.nn.Dropout(dropout)

        self.conv_2 = nn.Conv1d(
            encoder_embedding_size,
            encoder_embedding_size,
            kernel_size=encoder_kernel_size,
            stride=1,
            padding=int((encoder_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_2 = nn.BatchNorm1d(encoder_embedding_size)
        self.dropout_2 = torch.nn.Dropout(dropout)

        self.conv_3 = nn.Conv1d(
            encoder_embedding_size,
            encoder_embedding_size,
            kernel_size=encoder_kernel_size,
            stride=1,
            padding=int((encoder_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_3 = nn.BatchNorm1d(encoder_embedding_size)
        self.dropout_3 = torch.nn.Dropout(dropout)    

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder pre-network.
        
        Args:
            text: Input text tensor of shape (batch_size, seq_len) containing token indices
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_size)
        """
        # (N, S, E)
        x = self.embedding(text)
        # (N, S, E)
        x = self.linear_1(x)
        
        # (N, E, S)
        x = x.transpose(2, 1)

        # (N, E, S)
        x = self.conv_1(x)
        # (N, E, S)
        x = self.bn_1(x)
        # (N, E, S)
        x = F.relu(x)
        # (N, E, S)
        x = self.dropout_1(x)

        # (N, E, S)
        x = self.conv_2(x)
        # (N, E, S)
        x = self.bn_2(x)
        # (N, E, S)
        x = F.relu(x)
        # (N, E, S)
        x = self.dropout_2(x)

        # (N, E, S)
        x = self.conv_3(x)
        # (N, E, S)
        x = self.bn_3(x)
        # (N, E, S)
        x = F.relu(x)
        # (N, E, S)
        x = self.dropout_3(x)

        # (N, S, E)
        x = x.transpose(1, 2)
        # (N, S, E)
        x = self.linear_2(x)

        # (N, S, E)
        return x

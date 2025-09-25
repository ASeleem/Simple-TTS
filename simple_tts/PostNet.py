"""PostNet module for refining mel-spectrogram outputs in a TTS system.
"""
import torch
import torch.nn as nn


class PostNet(nn.Module):
    """PostNet for refining mel-spectrogram outputs in a TTS system."""
    def __init__(
        self,
        mel_freq: int,
        postnet_embedding_size: int,
        postnet_kernel_size: int,
        dropout: float = 0.5
    ):
        """
        Initialize PostNet with configurable parameters.
        
        Args:
            mel_freq: Number of mel frequency bins
            postnet_embedding_size: Size of the postnet hidden layers
            postnet_kernel_size: Kernel size for convolutional layers
            dropout: Dropout probability
        """
        super(PostNet, self).__init__()

        self.conv_1 = nn.Conv1d(
            mel_freq,
            postnet_embedding_size,
            kernel_size=postnet_kernel_size,
            stride=1,
            padding=int((postnet_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_1 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout_1 = torch.nn.Dropout(dropout)

        self.conv_2 = nn.Conv1d(
            postnet_embedding_size,
            postnet_embedding_size,
            kernel_size=postnet_kernel_size,
            stride=1,
            padding=int((postnet_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_2 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout_2 = torch.nn.Dropout(dropout)

        self.conv_3 = nn.Conv1d(
            postnet_embedding_size,
            postnet_embedding_size,
            kernel_size=postnet_kernel_size,
            stride=1,
            padding=int((postnet_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_3 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout_3 = torch.nn.Dropout(dropout)

        self.conv_4 = nn.Conv1d(
            postnet_embedding_size,
            postnet_embedding_size,
            kernel_size=postnet_kernel_size,
            stride=1,
            padding=int((postnet_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_4 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout_4 = torch.nn.Dropout(dropout)

        self.conv_5 = nn.Conv1d(
            postnet_embedding_size,
            postnet_embedding_size,
            kernel_size=postnet_kernel_size,
            stride=1,
            padding=int((postnet_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_5 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout_5 = torch.nn.Dropout(dropout)

        self.conv_6 = nn.Conv1d(
            postnet_embedding_size,
            mel_freq,
            kernel_size=postnet_kernel_size,
            stride=1,
            padding=int((postnet_kernel_size - 1) / 2),
            dilation=1
        )
        self.bn_6 = nn.BatchNorm1d(mel_freq)
        self.dropout_6 = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PostNet.
        
        Args:
            x: Input mel-spectrogram tensor of shape (batch_size, time_steps, mel_freq)
            
        Returns:
            Output tensor of shape (batch_size, time_steps, mel_freq) representing
            the residual to be added to the original mel-spectrogram
        """
        # x - (N, TIME, FREQ)
        # (N, FREQ, TIME)
        x = x.transpose(2, 1)

        # (N, POSNET_DIM, TIME)
        x = self.conv_1(x)
        # (N, POSNET_DIM, TIME)
        x = self.bn_1(x)
        # (N, POSNET_DIM, TIME)
        x = torch.tanh(x)
        # (N, POSNET_DIM, TIME)
        x = self.dropout_1(x)  

        # (N, POSNET_DIM, TIME)
        x = self.conv_2(x)
        # (N, POSNET_DIM, TIME)
        x = self.bn_2(x)
        # (N, POSNET_DIM, TIME)
        x = torch.tanh(x)
        # (N, POSNET_DIM, TIME)
        x = self.dropout_2(x)

        # (N, POSNET_DIM, TIME)
        x = self.conv_3(x)
        # (N, POSNET_DIM, TIME)
        x = self.bn_3(x)
        # (N, POSNET_DIM, TIME)
        x = torch.tanh(x)
        # (N, POSNET_DIM, TIME)
        x = self.dropout_3(x)

        # (N, POSNET_DIM, TIME)
        x = self.conv_4(x)
        # (N, POSNET_DIM, TIME)
        x = self.bn_4(x)
        # (N, POSNET_DIM, TIME)
        x = torch.tanh(x)
        # (N, POSNET_DIM, TIME)
        x = self.dropout_4(x)

        # (N, POSNET_DIM, TIME)
        x = self.conv_5(x)
        # (N, POSNET_DIM, TIME)
        x = self.bn_5(x)
        # (N, POSNET_DIM, TIME)
        x = torch.tanh(x)
        # (N, POSNET_DIM, TIME)
        x = self.dropout_5(x)

        # (N, FREQ, TIME)
        x = self.conv_6(x)
        # (N, FREQ, TIME)
        x = self.bn_6(x)
        # (N, FREQ, TIME)
        x = self.dropout_6(x)

        # (N, TIME, FREQ)
        x = x.transpose(1, 2)

        # (N, TIME, FREQ)
        return x

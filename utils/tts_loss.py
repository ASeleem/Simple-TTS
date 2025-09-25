"""
TTS Loss function for training text-to-speech models.
"""
import torch


class TTSLoss(torch.nn.Module):
    """
    TTS Loss function for training text-to-speech models.
    
    Based on: https://github.com/NVIDIA/tacotron2/blob/master/loss_function.py
    """
    def __init__(self, r_gate: float = 1.0):
        """
        Initialize TTSLoss with configurable parameters.
        
        Args:
            r_gate: Weight factor for the stop token loss
        """
        super(TTSLoss, self).__init__()

        self.r_gate = r_gate
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(
        self,
        mel_postnet_out: torch.Tensor,
        mel_out: torch.Tensor,
        stop_token_out: torch.Tensor,
        mel_target: torch.Tensor,
        stop_token_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for TTS loss calculation.
        
        Args:
            mel_postnet_out: Mel spectrogram output after postnet
            mel_out: Mel spectrogram output before postnet
            stop_token_out: Stop token predictions
            mel_target: Target mel spectrogram
            stop_token_target: Target stop tokens
            
        Returns:
            Combined loss (mel loss + weighted stop token loss)
        """
        stop_token_target = stop_token_target.view(-1, 1)
        stop_token_out = stop_token_out.view(-1, 1)

        mel_loss = self.mse_loss(mel_out, mel_target) + \
            self.mse_loss(mel_postnet_out, mel_target)

        stop_token_loss = self.bce_loss(stop_token_out, stop_token_target) * self.r_gate

        return mel_loss + stop_token_loss

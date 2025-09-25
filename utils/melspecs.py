"""
Utility functions for mel spectrogram processing.
"""
import torch
import torchaudio
from typing import Optional

from configs import Configs


def create_transforms(device: Optional[torch.device] = None):
    """
    Create audio transform objects with specified device.
    
    Args:
        device: Device to place transforms on. If None, uses CPU.
        
    Returns:
        Tuple of transform objects
    """
    if device is None:
        device = torch.device('cpu')

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=Configs.n_fft,
        win_length=Configs.win_length,
        hop_length=Configs.hop_length,
        power=Configs.power
    ).to(device)

    mel_scale_transform = torchaudio.transforms.MelScale(
        n_mels=Configs.mel_freq,
        sample_rate=Configs.sr,
        n_stft=Configs.n_stft
    ).to(device)

    mel_inverse_transform = torchaudio.transforms.InverseMelScale(
        n_mels=Configs.mel_freq,
        sample_rate=Configs.sr,
        n_stft=Configs.n_stft
    ).to(device)

    griffnlim_transform = torchaudio.transforms.GriffinLim(
        n_fft=Configs.n_fft,
        win_length=Configs.win_length,
        hop_length=Configs.hop_length
    ).to(device)

    return spec_transform, mel_scale_transform, mel_inverse_transform, griffnlim_transform


# Default CPU transforms for backward compatibility
spec_transform, mel_scale_transform, mel_inverse_transform, griffnlim_transform = create_transforms()


def norm_mel_spec_db(mel_spec):
    """Normalize mel spectrogram in dB scale."""
    mel_spec = ((2.0 * mel_spec - Configs.min_level_db) / (Configs.max_db / Configs.norm_db)) - 1.0
    mel_spec = torch.clip(mel_spec, -Configs.ref * Configs.norm_db, Configs.ref * Configs.norm_db)
    return mel_spec


def denorm_mel_spec_db(mel_spec):
    """Denormalize mel spectrogram from normalized dB scale."""
    mel_spec = (((1.0 + mel_spec) * (Configs.max_db / Configs.norm_db)) + Configs.min_level_db) / 2.0
    return mel_spec


def pow_to_db_mel_spec(mel_spec):
    """Convert power mel spectrogram to dB scale."""
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec,
        multiplier=Configs.ampl_multiplier,
        amin=Configs.ampl_amin,
        db_multiplier=Configs.db_multiplier,
        top_db=Configs.max_db
    )
    mel_spec = mel_spec / Configs.scale_db
    return mel_spec


def db_to_power_mel_spec(mel_spec):
    """Convert dB mel spectrogram to power scale."""
    mel_spec = mel_spec * Configs.scale_db
    mel_spec = torchaudio.functional.DB_to_amplitude(
        mel_spec,
        ref=Configs.ampl_ref,
        power=Configs.ampl_power
    )
    return mel_spec


def convert_to_mel_spec(wav: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert waveform to mel spectrogram in dB scale.
    
    Args:
        wav: Input waveform tensor
        device: Device to perform computation on. If None, uses wav's device.
        
    Returns:
        Mel spectrogram in dB scale
    """
    if device is None:
        device = wav.device

    # Create transforms on the appropriate device
    spec_trans, mel_scale_trans, _, _ = create_transforms(device)

    # Ensure wav is on the correct device
    wav = wav.to(device)

    spec = spec_trans(wav)
    mel_spec = mel_scale_trans(spec)
    db_mel_spec = pow_to_db_mel_spec(mel_spec)
    db_mel_spec = db_mel_spec.squeeze(0)
    return db_mel_spec


def inverse_mel_spec_to_wav(mel_spec: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert mel spectrogram back to waveform using Griffin-Lim.
    
    Args:
        mel_spec: Input mel spectrogram tensor
        device: Device to perform computation on. If None, uses mel_spec's device.
        
    Returns:
        Reconstructed waveform
    """
    if device is None:
        device = mel_spec.device

    # Create transforms on the appropriate device
    _, _, mel_inverse_trans, grifflim_trans = create_transforms(device)

    # Ensure mel_spec is on the correct device
    mel_spec = mel_spec.to(device)

    power_mel_spec = db_to_power_mel_spec(mel_spec)
    spectrogram = mel_inverse_trans(power_mel_spec)
    pseudo_wav = grifflim_trans(spectrogram)
    return pseudo_wav

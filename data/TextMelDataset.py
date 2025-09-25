"""Dataset and collate function for text and mel-spectrogram pairs.
"""

from typing import Tuple, List

import torch
import torchaudio
import pandas as pd

from utils.text_to_seq import text_to_seq
from utils.mask_from_seq_lengths import mask_from_seq_lengths
from utils.melspecs import convert_to_mel_spec


class TextMelDataset(torch.utils.data.Dataset):
    """Dataset for text and mel-spectrogram pairs.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        wav_path: str,
        sample_rate: int = 22050,
        use_cache: bool = True
    ):
        """
        Initialize TextMelDataset with configurable parameters.
        
        Args:
            df: DataFrame containing text and wav file information
            wav_path: Path to directory containing wav files
            sample_rate: Expected sample rate of audio files
            use_cache: Whether to cache processed items
        """
        self.df = df
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.use_cache = use_cache
        self.cache = {} if use_cache else None

    def get_item(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single row from the dataframe.
        
        Args:
            row: Row from the dataframe containing wav and text information
            
        Returns:
            Tuple of (text_tensor, mel_tensor)
        """
        wav_id = row["wav"]
        wav_path = f"{self.wav_path}/{wav_id}.wav"

        text = row["text_norm"]
        text = text_to_seq(text)

        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
        assert sample_rate == self.sample_rate, f"Expected {self.sample_rate} Hz, got {sample_rate} Hz"

        mel = convert_to_mel_spec(waveform, device=waveform.device)

        return (text, mel)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Tuple of (text_tensor, mel_tensor)
        """
        row = self.df.iloc[index]
        wav_id = row["wav"]

        if self.use_cache:
            text_mel = self.cache.get(wav_id)
            if text_mel is None:
                text_mel = self.get_item(row)
                self.cache[wav_id] = text_mel
        else:
            text_mel = self.get_item(row)

        return text_mel

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)


def text_mel_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Collate function for text-mel pairs.
    
    Args:
        batch: List of (text, mel) tuples
        
    Returns:
        Tuple containing:
            - texts_padded: Padded text sequences
            - text_lengths: Original text lengths
            - mels_padded: Padded mel spectrograms
            - mel_lengths: Original mel lengths
            - stop_token_padded: Stop token targets
    """
    text_length_max = torch.tensor(
        [text.shape[-1] for text, _ in batch],
        dtype=torch.int32
    ).max()

    mel_length_max = torch.tensor(
        [mel.shape[-1] for _, mel in batch],
        dtype=torch.int32
    ).max()

    text_lengths = []
    mel_lengths = []
    texts_padded = []
    mels_padded = []

    for text, mel in batch:
        text_length = text.shape[-1]

        text_padded = torch.nn.functional.pad(
            text,
            pad=[0, text_length_max - text_length],
            value=0
        )

        mel_length = mel.shape[-1]
        mel_padded = torch.nn.functional.pad(
            mel,
            pad=[0, mel_length_max - mel_length],
            value=0
        )

        text_lengths.append(text_length)
        mel_lengths.append(mel_length)
        texts_padded.append(text_padded)
        mels_padded.append(mel_padded)

    text_lengths = torch.tensor(text_lengths, dtype=torch.int32)
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.int32)
    texts_padded = torch.stack(texts_padded, 0)
    mels_padded = torch.stack(mels_padded, 0).transpose(1, 2)

    stop_token_padded = mask_from_seq_lengths(
        mel_lengths,
        mel_length_max
    )
    stop_token_padded = (~stop_token_padded).float()
    stop_token_padded[:, -1] = 1.0

    return (
        texts_padded,
        text_lengths,
        mels_padded,
        mel_lengths,
        stop_token_padded
    )

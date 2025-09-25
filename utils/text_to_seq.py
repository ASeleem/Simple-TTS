"""Convert text to a sequence of IDs."""
import torch
from typing import List, Optional
from configs import Configs


def create_symbol_to_id_mapping(symbols: List[str]) -> dict:
    """
    Create a mapping from symbols to IDs.
    
    Args:
        symbols: List of symbols
        
    Returns:
        Dictionary mapping symbols to their IDs
    """
    return {s: i for i, s in enumerate(symbols)}


def text_to_seq(
    text: str,
    symbols: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Convert text to a sequence of symbol IDs.
    
    Args:
        text: Input text string
        symbols: List of symbols to use. If None, uses Configs.symbols
        
    Returns:
        Tensor of symbol IDs with EOS token appended
    """
    if symbols is None:
        symbols = Configs.symbols

    symbol_to_id = create_symbol_to_id_mapping(symbols)

    text = text.lower()
    seq = []

    for char in text:
        char_id = symbol_to_id.get(char, None)
        if char_id is not None:
            seq.append(char_id)

    # Append EOS token
    if "EOS" in symbol_to_id:
        seq.append(symbol_to_id["EOS"])

    return torch.tensor(seq, dtype=torch.long)

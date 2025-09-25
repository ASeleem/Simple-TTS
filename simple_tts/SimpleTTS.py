"""A simple text-to-speech model using transformer architecture.
"""
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from simple_tts import EncoderBlock
from simple_tts import DecoderBlock
from simple_tts import EncoderPreNet
from simple_tts import DecoderPreNet
from simple_tts import PostNet
from utils.mask_from_seq_lengths import mask_from_seq_lengths


class SimpleTTS(nn.Module):
    """A simple text-to-speech model using transformer architecture.
    """
    def __init__(
        self,
        text_num_embeddings: int,
        embedding_size: int,
        encoder_embedding_size: int,
        mel_freq: int,
        max_mel_time: int,
        dim_feedforward: int,
        postnet_embedding_size: int,
        encoder_kernel_size: int,
        postnet_kernel_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_encoder_blocks: int = 3,
        num_decoder_blocks: int = 3,
    ):
        """
        Initialize SimpleTTS model with configurable parameters.
        
        Args:
            text_num_embeddings: Size of the text vocabulary
            embedding_size: Size of the transformer embeddings
            encoder_embedding_size: Size of the encoder pre-net embeddings
            mel_freq: Number of mel frequency bins
            max_mel_time: Maximum mel spectrogram time steps
            dim_feedforward: Size of the feedforward network
            postnet_embedding_size: Size of the postnet hidden layers
            encoder_kernel_size: Kernel size for encoder conv layers
            postnet_kernel_size: Kernel size for postnet conv layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_encoder_blocks: Number of encoder blocks
            num_decoder_blocks: Number of decoder blocks
            device: Device to run on
        """
        super(SimpleTTS, self).__init__()

        self.mel_freq = mel_freq
        self.max_mel_time = max_mel_time
        self.embedding_size = embedding_size

        self.encoder_prenet = EncoderPreNet(
            text_num_embeddings=text_num_embeddings,
            encoder_embedding_size=encoder_embedding_size,
            embedding_size=embedding_size,
            encoder_kernel_size=encoder_kernel_size,
            dropout=0.5
        )

        self.decoder_prenet = DecoderPreNet(
            mel_freq=mel_freq,
            embedding_size=embedding_size,
            dropout=0.5
        )

        self.postnet = PostNet(
            mel_freq=mel_freq,
            postnet_embedding_size=postnet_embedding_size,
            postnet_kernel_size=postnet_kernel_size,
            dropout=0.5
        )

        self.pos_encoding = nn.Embedding(
            num_embeddings=max_mel_time,
            embedding_dim=embedding_size
        )

        # Create encoder blocks dynamically
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                embedding_size=embedding_size,
                dim_feedforward=dim_feedforward,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_encoder_blocks)
        ])

        # Create decoder blocks dynamically
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embedding_size=embedding_size,
                dim_feedforward=dim_feedforward,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_decoder_blocks)
        ])

        self.linear_1 = nn.Linear(embedding_size, mel_freq)
        self.linear_2 = nn.Linear(embedding_size, 1)

        self.norm_memory = nn.LayerNorm(normalized_shape=embedding_size)


    def forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        mel: torch.Tensor,
        mel_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SimpleTTS model.
        
        Args:
            text: Input text tensor of shape (batch_size, text_seq_len)
            text_len: Text sequence lengths tensor of shape (batch_size,)
            mel: Input mel spectrogram tensor of shape (batch_size, mel_time_steps, mel_freq)
            mel_len: Mel sequence lengths tensor of shape (batch_size,)
            
        Returns:
            Tuple containing:
                - mel_postnet: Mel spectrogram with postnet applied
                - mel_linear: Mel spectrogram before postnet
                - stop_token: Stop token predictions
        """

        N = text.shape[0]
        S = text.shape[1]
        TIME = mel.shape[1]

        self.src_key_padding_mask = torch.zeros(
            (N, S),
            device=text.device
        ).masked_fill(
          ~mask_from_seq_lengths(
            text_len,
            max_length=S
          ),
          float("-inf")
        )

        self.src_mask = torch.zeros(
          (S, S),
          device=text.device
        ).masked_fill(
          torch.triu(
              torch.full(
                  (S, S),
                  True,
                  dtype=torch.bool
              ),
              diagonal=1
          ).to(text.device),
          float("-inf")
        )

        self.tgt_key_padding_mask = torch.zeros(
          (N, TIME),
          device=mel.device
        ).masked_fill(
          ~mask_from_seq_lengths(
            mel_len,
            max_length=TIME
          ),
          float("-inf")
        )

        self.tgt_mask = torch.zeros(
          (TIME, TIME),
          device=mel.device
        ).masked_fill(
          torch.triu(
              torch.full(
                  (TIME, TIME),
                  True,
                  device=mel.device,
                  dtype=torch.bool
              ),
              diagonal=1
          ),       
          float("-inf")
        )

        self.memory_mask = torch.zeros(
          (TIME, S),
          device=mel.device
        ).masked_fill(
          torch.triu(
              torch.full(
                  (TIME, S),
                  True,
                  device=mel.device,
                  dtype=torch.bool
              ),
              diagonal=1
          ),
          float("-inf")
        )

        # (N, S, E)
        text_x = self.encoder_prenet(text)

        # (MAX_S_TIME, E)
        pos_codes = self.pos_encoding(
            torch.arange(self.max_mel_time).to(mel.device)
        )

        S = text_x.shape[1]
        text_x = text_x + pos_codes[:S]
        # dropout after pos encoding?

        # Pass through encoder blocks
        # (N, S, E)
        for encoder_block in self.encoder_blocks:
            text_x = encoder_block(
                text_x,
                attn_mask=self.src_mask,
                key_padding_mask=self.src_key_padding_mask
            )

        text_x = self.norm_memory(text_x)

        # (N, TIME, E)
        mel_x = self.decoder_prenet(mel)  
        mel_x = mel_x + pos_codes[:TIME]
        # dropout after pos encoding?

        # Pass through decoder blocks
        # (N, TIME, E)
        for decoder_block in self.decoder_blocks:
            mel_x = decoder_block(
                x=mel_x,
                memory=text_x,
                x_attn_mask=self.tgt_mask,
                x_key_padding_mask=self.tgt_key_padding_mask,
                memory_attn_mask=self.memory_mask,
                memory_key_padding_mask=self.src_key_padding_mask
            )

        # (N, TIME, FREQ)
        mel_linear = self.linear_1(mel_x)
        # (N, TIME, FREQ)
        mel_postnet = self.postnet(mel_linear)
        # (N, TIME, FREQ)
        mel_postnet = mel_linear + mel_postnet
        # (N, TIME, 1)
        stop_token = self.linear_2(mel_x)

        bool_mel_mask = self.tgt_key_padding_mask.ne(0).unsqueeze(-1).repeat(
            1, 1, self.mel_freq
        )

        mel_linear = mel_linear.masked_fill(
          bool_mel_mask,
          0
        )
        mel_postnet = mel_postnet.masked_fill(
          bool_mel_mask,
          0
        )

        stop_token = stop_token.masked_fill(
          bool_mel_mask[:, :, 0].unsqueeze(-1),
          1e3
        ).squeeze(2)

        return mel_postnet, mel_linear, stop_token


    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        max_length: int = 800,
        stop_token_threshold: float = 0.5,
        with_tqdm: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mel spectrogram from text using inference.
        
        Args:
            text: Input text tensor of shape (1, text_seq_len)
            max_length: Maximum generation length
            stop_token_threshold: Threshold for stopping generation
            with_tqdm: Whether to show progress bar
            
        Returns:
            Tuple containing:
                - mel_postnet: Generated mel spectrogram
                - stop_token_outputs: Stop token predictions
        """
        self.eval()
        self.train(False)
        device = text.device
        text_lengths = torch.tensor(text.shape[1]).unsqueeze(0).to(device)
        N = 1
        SOS = torch.zeros((N, 1, self.mel_freq), device=device)

        mel_padded = SOS
        mel_lengths = torch.tensor(1).unsqueeze(0).to(device)
        stop_token_outputs = torch.FloatTensor([]).to(device)

        if with_tqdm:
            iters = tqdm(range(max_length))
        else:
            iters = range(max_length)

        for _ in iters:
            mel_postnet, mel_linear, stop_token = self(
                text,
                text_lengths,
                mel_padded,
                mel_lengths
            )

            mel_padded = torch.cat(
                [
                    mel_padded,
                    mel_postnet[:, -1:, :]
                ],
                dim=1
            )
            if torch.sigmoid(stop_token[:, -1]) > stop_token_threshold:
                break
            else:
                stop_token_outputs = torch.cat([stop_token_outputs, stop_token[:, -1:]], dim=1)
                mel_lengths = torch.tensor(mel_padded.shape[1]).unsqueeze(0).to(device)

        return mel_postnet, stop_token_outputs

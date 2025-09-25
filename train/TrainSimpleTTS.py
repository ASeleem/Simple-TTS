"Training class for SimpleTTS model."
import os
import time
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from simple_tts import SimpleTTS
from configs.Configs import Configs
from data import TextMelDataset, text_mel_collate_fn
from utils.tts_loss import TTSLoss
from utils.melspecs import inverse_mel_spec_to_wav
from utils.text_to_seq import text_to_seq


class TrainSimpleTTS:
    """
    Training class for SimpleTTS model.
    """

    def __init__(self, config: Optional[Configs] = None, device: Optional[str] = None):
        """
        Initialize TrainSimpleTTS with configuration.
        
        Args:
            config: Configuration object. If None, uses default Configs()
            device: CUDA device specification (e.g., "cuda:0", "cuda:1", "cpu"). 
                   If None, uses "cuda:0" if CUDA is available, otherwise "cpu"
        """
        self.config = config if config is not None else Configs()

        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize training state
        self.model: Optional[SimpleTTS] = None
        self.criterion: Optional[TTSLoss] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.logger: Optional[SummaryWriter] = None

        # Training metrics
        self.best_test_loss_mean = float("inf")
        self.best_train_loss_mean = float("inf")
        self.train_loss_mean = 0.0
        self.epoch = 0
        self.step = 0

    def _batch_process(self, batch: Tuple) -> Tuple[torch.Tensor, ...]:
        """
        Process a batch for training/testing.
        
        Args:
            batch: Input batch from dataloader
            
        Returns:
            Tuple of processed tensors
        """
        text_padded, text_lengths, mel_padded, mel_lengths, stop_token_padded = batch

        text_padded = text_padded.to(self.device)
        text_lengths = text_lengths.to(self.device)
        mel_padded = mel_padded.to(self.device)
        stop_token_padded = stop_token_padded.to(self.device)
        mel_lengths = mel_lengths.to(self.device)

        N = mel_padded.shape[0]
        SOS = torch.zeros((N,
                           1,
                           self.config.mel_freq),
                          device=mel_padded.device)  # Start of sequence

        mel_input = torch.cat(
            [
                SOS,
                mel_padded[:, :-1, :]  # (N, L, FREQ)
            ],
            dim=1
        )

        return (
            text_padded,
            text_lengths,
            mel_padded,
            mel_lengths,
            mel_input,
            stop_token_padded
        )

    def _inference_utterance(self, text: str) -> Tuple[torch.Tensor, plt.Figure]:
        """
        Generate audio and visualization from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio tensor, matplotlib figure)
        """
        sequences = text_to_seq(text).unsqueeze(0).to(self.device)
        postnet_mel, stop_token = self.model.inference(
            sequences,
            stop_token_threshold=1e5,
            with_tqdm=False
        )
        audio = inverse_mel_spec_to_wav(postnet_mel.detach()[0].T, self.device)

        fig, (ax1) = plt.subplots(1, 1)
        ax1.imshow(
            postnet_mel[0, :, :].detach().cpu().numpy().T,
        )

        return audio, fig

    def _calculate_test_loss(self, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Calculate test loss on the test dataset.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Average test loss
        """
        test_loss_mean = 0.0
        self.model.eval()

        with torch.no_grad():
            for test_i, test_batch in enumerate(test_loader):
                test_text_padded, test_text_lengths, test_mel_padded, test_mel_lengths, \
                test_mel_input, test_stop_token_padded = self._batch_process(test_batch)

                test_post_mel_out, test_mel_out, test_stop_token_out = self.model(
                    test_text_padded,
                    test_text_lengths,
                    test_mel_input,
                    test_mel_lengths
                )
                test_loss = self.criterion(
                    mel_postnet_out=test_post_mel_out,
                    mel_out=test_mel_out,
                    stop_token_out=test_stop_token_out,
                    mel_target=test_mel_padded,
                    stop_token_target=test_stop_token_padded
                )

                test_loss_mean += test_loss.item()

        test_loss_mean = test_loss_mean / (test_i + 1)
        return test_loss_mean

    def train(self):
        """
        Main training method that implements the complete training loop.
        """
        torch.manual_seed(self.config.seed)

        df = pd.read_csv(self.config.csv_path)
        train_df, test_df = train_test_split(
            df,
            test_size=64,
            random_state=self.config.seed
        )

        # Create datasets with configuration
        train_dataset = TextMelDataset(
            df=train_df,
            wav_path=self.config.wav_path,
            sample_rate=self.config.sr
        )
        test_dataset = TextMelDataset(
            df=test_df,
            wav_path=self.config.wav_path,
            sample_rate=self.config.sr
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=2,
            shuffle=True,
            sampler=None,
            batch_size=self.config.batch_size,
            pin_memory=True,
            drop_last=True,
            collate_fn=text_mel_collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=2,
            shuffle=True,
            sampler=None,
            batch_size=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=text_mel_collate_fn
        )

        train_saved_path = f"{self.config.save_path}/train_{self.config.save_name}"
        test_saved_path = f"{self.config.save_path}/test_{self.config.save_name}"

        print("train_saved_path:", train_saved_path)
        print("test_saved_path:", test_saved_path)

        self.logger = SummaryWriter(self.config.log_path)
        self.criterion = TTSLoss(r_gate=self.config.r_gate).to(self.device)

        # Initialize model with configuration parameters
        self.model = SimpleTTS(
            text_num_embeddings=self.config.text_num_embeddings,
            embedding_size=self.config.embedding_size,
            encoder_embedding_size=self.config.encoder_embedding_size,
            mel_freq=self.config.mel_freq,
            max_mel_time=self.config.max_mel_time,
            dim_feedforward=self.config.dim_feedforward,
            postnet_embedding_size=self.config.postnet_embedding_size,
            encoder_kernel_size=self.config.encoder_kernel_size,
            postnet_kernel_size=self.config.postnet_kernel_size
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.scaler = torch.cuda.amp.GradScaler()

        if os.path.isfile(train_saved_path):
            state = torch.load(train_saved_path)
            state_model = state["model"]
            state_optimizer = state["optimizer"]

            self.step = state["i"] + 1
            self.best_test_loss_mean = state.get("test_loss", float("inf"))
            self.best_train_loss_mean = state.get("train_loss", float("inf"))

            self.model.load_state_dict(state_model)
            self.optimizer.load_state_dict(state_optimizer)

            print(f"Load: {self.step}; test_loss: {np.round(self.best_test_loss_mean, 5)}; train_loss: {np.round(self.best_train_loss_mean, 5)}")
        else:
            print("Start from zero!")

        start_time_sec = time.time()
        while True:
            for batch in train_loader:
                text_padded, text_lengths, mel_padded, mel_lengths, mel_input, stop_token_padded = self._batch_process(batch)

                self.model.train(True)
                self.model.zero_grad()

                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    post_mel_out, mel_out, stop_token_out = self.model(
                        text_padded,
                        text_lengths,
                        mel_input,
                        mel_lengths
                    )
                    loss = self.criterion(
                        mel_postnet_out=post_mel_out,
                        mel_out=mel_out,
                        stop_token_out=stop_token_out,
                        mel_target=mel_padded,
                        stop_token_target=stop_token_padded
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.train_loss_mean += loss.item()

                if self.step != 0 and self.step % self.config.step_print == 0:
                    self.train_loss_mean = self.train_loss_mean / self.config.step_print
                    self.logger.add_scalar("Loss/train_loss", self.train_loss_mean, global_step=self.step)

                    if self.step % self.config.step_test == 0:
                        test_loss_mean = self._calculate_test_loss(test_loader)
                        audio, fig = self._inference_utterance("Hello, World.")

                        self.logger.add_scalar("Loss/test_loss", test_loss_mean, global_step=self.step)
                        self.logger.add_figure(f"Img/img_{self.step}", fig, global_step=self.step)
                        self.logger.add_audio(f"Utterance/audio_{self.step}", audio, sample_rate=self.config.sr, global_step=self.step)

                        print(f"{self.epoch}-{self.step}) Test loss: {np.round(test_loss_mean, 5)}")

                        if self.step % self.config.step_save == 0:
                            is_best_train = self.train_loss_mean < self.best_train_loss_mean
                            is_best_test = test_loss_mean < self.best_test_loss_mean

                            state = {
                                "model": self.model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "i": self.step,
                                "test_loss": test_loss_mean,
                                "train_loss": self.train_loss_mean
                            }

                            if is_best_train:
                                print(f"{self.epoch}-{self.step}) Save best train")
                                torch.save(state, train_saved_path)
                                self.best_train_loss_mean = self.train_loss_mean

                            if is_best_test:
                                print(f"{self.epoch}-{self.step}) Save best test")
                                torch.save(state, test_saved_path)
                                self.best_test_loss_mean = test_loss_mean

                    end_time_sec = time.time()
                    time_sec = np.round(end_time_sec - start_time_sec, 3)
                    start_time_sec = end_time_sec

                    print(f"{self.epoch}-{self.step}) Train loss: {np.round(self.train_loss_mean, 5)}; Duration: {time_sec} sec.")
                    self.train_loss_mean = 0.0

                self.step += 1
            self.epoch += 1

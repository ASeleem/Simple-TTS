# Simple TTS

**Transformer-based Text-to-Speech with complete training pipeline**

A clean PyTorch implementation of a transformer-based text-to-speech system featuring encoder-decoder architecture, mel-spectrogram generation, and Griffin-Lim vocoding. Based on the Neural Speech Synthesis with Transformer Network (Li et al., 2019). Includes a complete training pipeline, multi-GPU support, and production-ready inference.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Simple-TTS

# Install dependencies
pip install torch torchaudio tensorboard pandas matplotlib scikit-learn tqdm pydub
```

### Dataset Preparation

The system is designed for the LJSpeech dataset:

```bash
# Download LJSpeech dataset
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2

# Update paths in configs/Configs.py
csv_path = "/path/to/LJSpeech-1.1/metadata.csv"
wav_path = "/path/to/LJSpeech-1.1/wavs"
```

### Training

```python
from configs.Configs import Configs
from train import TrainSimpleTTS

# Initialize configuration and trainer
config = Configs()
trainer = TrainSimpleTTS(config, device="cuda:0")

# Start training
trainer.train()
```

Or use the provided training script:

```bash
python train.py
```

### Inference

```python
import torch
from configs.Configs import Configs
from simple_tts import SimpleTTS
from utils.text_to_seq import text_to_seq
from utils.melspecs import inverse_mel_spec_to_wav
from utils.write_mp3 import write_mp3

# Load configuration and model
config = Configs()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SimpleTTS(
    text_num_embeddings=config.text_num_embeddings,
    embedding_size=config.embedding_size,
    encoder_embedding_size=config.encoder_embedding_size,
    mel_freq=config.mel_freq,
    max_mel_time=config.max_mel_time,
    dim_feedforward=config.dim_feedforward,
    postnet_embedding_size=config.postnet_embedding_size,
    encoder_kernel_size=config.encoder_kernel_size,
    postnet_kernel_size=config.postnet_kernel_size
).to(device)

# Load trained weights
state = torch.load("path/to/model.pt", map_location=device)
model.load_state_dict(state["model"])

# Generate speech
text = "Hello, world! This is a test of the Simple TTS system."
mel_spec, stop_tokens = model.inference(
    text_to_seq(text).unsqueeze(0).to(device),
    max_length=200,
    stop_token_threshold=0.7
)

# Convert to audio
audio = inverse_mel_spec_to_wav(mel_spec[0].T, device)
write_mp3(audio.cpu().numpy(), "output.mp3")
```

## Architecture

### Model Components

- **EncoderPreNet**: Text embedding + 3-layer CNN with batch normalization
- **EncoderBlock**: Multi-head self-attention + feedforward network
- **DecoderPreNet**: Mel-spectrogram preprocessing with linear layers
- **DecoderBlock**: Self-attention + cross-attention + feedforward network
- **PostNet**: 5-layer CNN for mel-spectrogram refinement

### Training Pipeline

- **TextMelDataset**: Configurable dataset with caching support
- **TTSLoss**: Combined MSE (mel) + BCE (stop token) loss
- **TrainSimpleTTS**: Complete training class with mixed precision
- **Device Support**: Multi-GPU training with configurable device placement

## Configuration

All hyperparameters are centralized in `configs/Configs.py`:

```python
class Configs:
    # Audio parameters
    sr = 22050
    n_fft = 2048
    mel_freq = 128
    
    # Model parameters
    embedding_size = 256
    encoder_embedding_size = 512
    dim_feedforward = 1024
    
    # Training parameters
    batch_size = 32
    lr = 2e-4
    grad_clip = 1.0
```

## Notebooks

- `model_development.ipynb`: Model inference and visualization
- `MelSpectrogram_and_STFT.ipynb`: Audio processing pipeline analysis
- `GriffinLim_algorithm.ipynb`: Griffin-Lim vocoding demonstration

## Project Structure

```
Simple-TTS/
├── configs/
│   └── Configs.py              # Configuration parameters
├── simple_tts/
│   ├── EncoderBlock.py         # Transformer encoder
│   ├── DecoderBlock.py         # Transformer decoder
│   ├── EncoderPreNet.py        # Text preprocessing
│   ├── DecoderPreNet.py        # Mel preprocessing
│   ├── PostNet.py              # Mel refinement
│   └── SimpleTTS.py            # Main model
├── train/
│   └── TrainSimpleTTS.py       # Training pipeline
├── data/
│   └── TextMelDataset.py       # Dataset and collate function
├── utils/
│   ├── melspecs.py             # Mel-spectrogram processing
│   ├── text_to_seq.py          # Text preprocessing
│   ├── tts_loss.py             # Loss function
│   ├── write_mp3.py            # Audio export
│   └── mask_from_seq_lengths.py # Attention masking
├── train.py                    # Training entry point
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchaudio
- tensorboard
- pandas
- matplotlib
- scikit-learn
- tqdm
- pydub

## Training Tips

1. **Start Small**: Begin with a subset of the dataset to verify the pipeline
2. **Monitor Attention**: Check attention alignments in TensorBoard
3. **Stop Token**: Adjust `stop_token_threshold` for optimal sequence length
4. **Learning Rate**: Use warmup and decay for stable training
5. **Mixed Precision**: Enabled by default for faster training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper type hints and documentation
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Neural Speech Synthesis with Transformer Network (Li et al., 2019)
- Inspired by Tacotron2 for TTS-specific components
- Griffin-Lim algorithm for vocoding
- LJSpeech dataset for training and evaluation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2019neural,
  title={Neural Speech Synthesis with Transformer Network},
  author={Li, Naihan and Liu, Shujie and Liu, Yanqing and Zhao, Sheng and Liu, Ming and Zhou, Ming},
  journal={arXiv preprint arXiv:1809.08895},
  year={2019}
}

@misc{simple-tts,
  title={Simple TTS: Transformer-based Text-to-Speech System},
  author={Abdelrahman Seleem},
  year={2025},
  url={https://github.com/ASeleem/Simple-TTS}
}
```
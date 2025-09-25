# Import
from configs import Configs
from train import TrainSimpleTTS

# Train
config = Configs()
trainer = TrainSimpleTTS(config, device="cuda:3")
trainer.train()

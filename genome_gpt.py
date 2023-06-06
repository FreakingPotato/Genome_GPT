# minGPT code adopted from https://github.com/karpathy/minGPT

import numpy as np
import torch
import torch.nn as nn
import logging
from torch.nn import functional as F
from mingpt.utils import set_seed, data_cleaning, CharDataset
import math
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig

## set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

## make deterministic
set_seed(42)

# get the subset of the training data
train_sequence = open('data/hg38.500m.100m.10m.aa', 'r').read()
test_sequence = open('data/hg38.500m.100m.10m.ab', 'r').read()

block_size = 6 # spatial extent of the model for its context
train_dataset = CharDataset(data_cleaning(train_sequence), block_size)
test_dataset = CharDataset(data_cleaning(test_sequence), block_size)

## defining the GPT model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=128)
model = GPT(mconf)

batch_size = 2048 * 4

## initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=1, batch_size=batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                      num_workers=24)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
trainer.train()

## saving the final model
torch.save(model, "./saved_model.pth")
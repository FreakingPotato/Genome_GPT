# minGPT code adopted from https://github.com/karpathy/minGPT
import argparse
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.nn import functional as F
from mingpt.utils import set_seed, data_cleaning, CharDataset
import math
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig

def get_args():
        parser = argparse.ArgumentParser(description="Genome_GPT")
        parser.add_argument('-n_blocks', '--block_size', default=128, type=int,
                        help='the maximum sequence length')
        parser.add_argument('-n_heads', '--num_heads', default=8, type=int,
                        help='number of heads for MSA')
        parser.add_argument('-n_layers', '--num_layers', default=8, type=int,
                        help='number of layers for transformer')
        parser.add_argument('-n_embeds', '--num_embeddings', default=128, type=int,
                        help='number of embeddings for SA')
        parser.add_argument('-n_epochs', '--num_epochs', default=5, type=int,
                        help='peak calling method')
        parser.add_argument('-v', '--vocab_size', default=5, type=int,
                        help='vocabulary size genome sequences')
        parser.add_argument('-b', '--batch_size', default=512, type=int,
                        help='number of samples in each batch (default batch_size = 512)')
        parser.add_argument('-lr', '--learning_rate', default=6e-4, type=float,
                        help='learning rate default 6e-4')
        parser.add_argument('-lr_d', '--learning_rate_decay', default=True, type=bool,
                        help='learning rate decay default True')                
        parser.add_argument('-r', '--random_seed', default=42, type=int,
                        help='fix random seed')
        parser.add_argument('-f', '--path', default='./', type=str,
                        help='path of the model')
        args = parser.parse_args()
        return args


def main():
        args = get_args()
        n_block = args.block_size
        n_heads = args.num_heads
        n_layers = args.num_layers
        n_embeds = args.num_embeddings
        n_epochs = args.num_epochs
        vocab_size = args.vocab_size
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        learning_rate_decay = args.learning_rate_decay
        random_seed = args.random_seed
        model_path = args.path
        print(args)

        ## set up logging
        logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
        )

        ## make deterministic
        set_seed(random_seed)

        # get the subset of the training data
        train_sequence = open('data/hg38.500m.100m.10m.aa', 'r').read()
        test_sequence = open('data/hg38.500m.100m.10m.ab', 'r').read()

        block_size = n_block # spatial extent of the model for its context
        train_dataset = CharDataset(data_cleaning(train_sequence), block_size)
        test_dataset = CharDataset(data_cleaning(test_sequence), block_size)

        ## defining the GPT model
        mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                        n_layer=n_layers, n_head=n_heads, n_embd=n_embeds)
        model = GPT(mconf)

        ## initialize a trainer instance and kick off training
        tconf = TrainerConfig(max_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                        lr_decay=learning_rate_decay, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                        num_workers=24)
        trainer = Trainer(model, train_dataset, test_dataset, tconf)
        trainer.train()

        ## saving the final model
        torch.save(model, model_path+"saved_model.pth")

if __name__ == '__main__':
        main()
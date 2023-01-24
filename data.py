import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from config import Config


class Data:
    def __init__(self, config: Config):
        self.config = config
        self.text = ""
        self.X = []
        self.Y = []
        self.char_idx = []
        self.idx_char = []
        self.read_data()

    def encode_dataset(self):
        print("Encoding dataset...")
        vocab = sorted(list(set(self.text)))
        self.char_idx = {c: i for i, c in enumerate(sorted(vocab))}
        self.idx_char = {i: c for i, c in enumerate(sorted(vocab))}

        sequences = []
        targets = []

        # The next character in a sequence is the target
        for i in range(0, len(self.text) - self.config.input_size):
            sequences.append(self.text[i: i + self.config.input_size])
            targets.append(self.text[i + self.config.input_size])

        self.X = [[self.char_idx[c] for c in seq] for seq in sequences]
        self.Y = [[self.char_idx[c] for c in t] for t in targets]

        print("Text total length: {:,}".format(len(self.text)))
        print("Distinct chars   : {:,}".format(len(vocab)))
        print("Total sequences  : {:,}".format(len(sequences)))

    def read_data(self):
        with open(self.config.train_data_path, 'r') as fd:
            self.text = fd.readline()
            self.encode_dataset()

    def encode_text(self, input_string: str):
        enc = [self.char_idx[c] for c in input_string]
        return enc

    def decode_text(self, encoded_string: []):
        dec = [self.idx_char[c] for c in encoded_string]
        return dec

    def get_train_and_valid_data(self):
        return self.X, self.Y

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from config import Config

example = [
    "TakeOnMe:d=4,o=5,b=160:8e6,8e,8c,16a,8p.,16a,8p.,16d6,8p.,16d,8p.,8d,8f#,8f#\
    ,8g,8a,8g,8g,8g,16d,8p.,16c,8p.,16e,8p.,16e,8p.,8e,8d,8d,8e,8d,8e,8e,8c,16a,8p.\
    ,16a,8p.,16d6,8p.,16d,8p.,8d,8f#,8f#,8g,16a"]


class Data:
    def __init__(self, config: Config):
        self.config = config
        self.text = ""
        self.X = []
        self.Y = []
        self.char_idx = []
        self.idx_char = []
        self.vocab = []
        self.read_data()

    def encode_dataset(self):
        print("Encoding dataset...")
        self.vocab = sorted(list(set(self.text)))
        self.char_idx = {c: i for i, c in enumerate(sorted(self.vocab))}
        self.idx_char = {i: c for i, c in enumerate(sorted(self.vocab))}

        #print(self.vocab)

        sequences = []
        targets = []

        window = self.config.window_size

        # Generate targets using a sliding windows on the input
        for i in range(0, len(self.text) - self.config.input_size):
            sequences.append(self.text[i: i + window])
            targets.append(self.text[i + 1: i + window + 1])

        self.X = [[self.char_idx[c] for c in seq] for seq in sequences]
        self.Y = [[self.char_idx[c] for c in t] for t in targets]

        print("Text total length: {:,}".format(len(self.text)))
        print("Distinct chars   : {:,}".format(len(self.vocab)))
        print("Total sequences  : {:,}".format(len(sequences)))

    def read_data(self):
        with open(self.config.train_data_path, 'r') as fd:
            self.text = fd.readlines()
            self.encode_dataset()

    def encode_text(self, input_string: str):
        enc = [self.char_idx[c] for c in input_string]
        return enc

    def decode_text(self, encoded_string: []):
        dec = [self.idx_char[c] for c in encoded_string]
        return dec

    def get_train_and_valid_data(self):
        return self.X, self.Y

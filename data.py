import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from config import Config


def string_to_semi_redundant_sequences(text: str, seq_maxlen=Config.input_size, redun_step=1, char_idx=None):
    print("Vectorizing text...")

    chars = sorted(list(set(text)))

    if char_idx is None:
        char_idx = {c: i for i, c in enumerate(sorted(chars))}
        idx_char = {i: c for i, c in enumerate(sorted(chars))}

    len_chars = len(char_idx)

    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_maxlen, redun_step):
        sequences.append(text[i: i + seq_maxlen])
        next_chars.append(text[i + seq_maxlen])

    X = np.zeros((len(sequences), seq_maxlen, len_chars), dtype=np.bool)
    Y = np.zeros((len(sequences), len_chars), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_idx[char]] = 1
        Y[i, char_idx[next_chars[i]]] = 1
    #        print(".")

    print("Text total length: {:,}".format(len(text)))
    print("Distinct chars   : {:,}".format(len_chars))
    print("Total sequences  : {:,}".format(len(sequences)))

    return X, Y, char_idx


class Data:
    def __init__(self, config: Config):
        self.config = config
        self.vocab = None
        self.raw_text = self.read_data()
        self.X, self.Y, self.char_idx, self.idx_char = self.encode_dataset()

    def read_data(self):
        with open(self.config.train_data_path, 'r') as fd:
            init_data = fd.read()
        return init_data

    def encode_dataset(self):
        print("One_hot encoding dataset...")

        self.vocab = sorted(list(set(self.raw_text)))

        char_idx = {c: i for i, c in enumerate(sorted(self.vocab))}
        idx_char = {i: c for i, c in enumerate(sorted(self.vocab))}

        len_chars = len(char_idx)

        sequences = []
        next_chars = []
        for i in range(0, len(self.raw_text) - 1, 1):
            sequences.append(self.raw_text[i: i + 1])
            next_chars.append(self.raw_text[i + 1])

        X = np.zeros((len(sequences), 1, len_chars), dtype=np.bool)
        Y = np.zeros((len(sequences), len_chars), dtype=np.bool)
        for i, seq in enumerate(sequences):
            for t, char in enumerate(seq):
                X[i, t, char_idx[char]] = 1
            Y[i, char_idx[next_chars[i]]] = 1
        #        print(".")

        print(f"Text total length: {len(self.raw_text)}")
        print(f"Vocabulary length: {len_chars}")
        print(f"Vocabulary: {self.vocab}")
        print(f"Total sequences: {len(sequences)}")
        print(f"{len(X):}, {len(Y):}")

        return X, Y, char_idx, idx_char

    def encode_text(self, input_string: str):
        enc = [self.char_idx[c] for c in input_string]
        return enc

    def decode_text(self, encoded_string: []):
        dec = [self.idx_char[c] for c in encoded_string]
        return dec

    def get_train_and_valid_data(self):
        return self.X, self.Y

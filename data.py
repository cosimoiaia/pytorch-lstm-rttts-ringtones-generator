import pandas as pd
import numpy as np
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
        self.raw_data, self.X, self.Y, self.char_idx = self.read_data()

    def read_data(self):
        with open(self.config.train_data_path, 'r') as fd:
            init_data = fd.readline()
            X, Y, char_idx = string_to_semi_redundant_sequences(init_data)

        return init_data, X, Y, char_idx

    def get_train_and_valid_data(self):
        train_x, valid_x, train_y, valid_y = train_test_split(self.X, self.Y,
                                                              test_size=self.config.valid_data_rate)
                                                              #random_state=self.config.random_seed,
                                                              #shuffle=self.config.shuffle_train_data)
        #return [train_x, valid_x, train_y, valid_y]
        return [self.X, self.Y, self.X, self.Y]

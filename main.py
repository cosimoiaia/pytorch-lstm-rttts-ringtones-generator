# from model.model_pytorch import train, predict
from config import Config
from lstm_model import LSTM_Model, do_train
from data import Data
import logging

if __name__ == "__main__":
    config = Config()
    data = Data(config)
    model = do_train(config, logging.getLogger(), data.get_train_and_valid_data())


from config import Config
from lstm_model import LSTM_Model, do_train, predict
from data import Data

if __name__ == "__main__":
    config = Config()
    data = Data(config)
    model = do_train(config, data.get_train_and_valid_data())
    encoded = data.encode_text("alamarene:")
    predict(config, encoded)


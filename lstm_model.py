import logging

import torch
from torch.nn import Module, LSTM, Linear, Embedding
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.autograd import Variable
import numpy as np

from config import Config

logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)


class LSTM_Model(Module):
    """
    Basic wrapper for the pytorch implementation of LSTM
    """

    def __init__(self, config: Config, device):
        super(LSTM_Model, self).__init__()
        self.config = config
        self.device = device

        self.hidden = (Variable(torch.zeros(self.config.lstm_layers,
                                            self.config.batch_size, self.config.hidden_size)).to(device),
                       Variable(torch.zeros(self.config.lstm_layers,
                                            self.config.batch_size,self.config.hidden_size)).to(device))

        self.embedding = Embedding(config.input_size, config.hidden_size)
        self.lstm = LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, x):
        inputs = self.embedding(x)
        lstm_out, (h, c) = self.lstm(inputs, self.hidden)
        linear_out = self.linear(lstm_out.view[:, -1, :])
        return linear_out


def do_train(config: Config, train_and_valid_data):
    """
    Perform the training of the model and saves it.

    :param config: config class containing parameters
    :param logging: global logging
    :param train_and_valid_data: training and validation set
    :return: The trained model
    """

    X, Y = train_and_valid_data
    dataset = TensorDataset(torch.tensor(X, dtype=torch.long),
                            torch.tensor(Y, dtype=torch.long))
    train_size = int(config.train_data_rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size)  # DataLoader

    valid_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = LSTM_Model(config, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0

    for epoch in range(config.epoch):
        model.train()
        train_loss_array = []

        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            optimizer.zero_grad()
            print(f"{_train_X.shape:}")
            pred_y = model(_train_X)

            loss = criterion(pred_y, _train_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step += 1

        model.eval()
        valid_loss_array = []

        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_y = model(_valid_X)
            loss = criterion(pred_y, _valid_Y)
            valid_loss_array.append(loss.item())

        valid_loss = sum(valid_loss_array) / len(valid_loss_array)
        train_loss = sum(train_loss_array) / len(train_loss_array)

        print("The train loss is {:.6f}. ".format(train_loss) +
              "The valid loss is {:.6f}.".format(valid_loss))

        if valid_loss <= valid_loss_min:
            bad_epoch = 0
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), config.model_save_path + config.model_name)
        else:
            bad_epoch += 1
            if bad_epoch >= 5:
                print(" The training stops early in epoch {}".format(epoch))
                break

    return model


def predict(config: Config, data: np.array):
    """
    Return predictions on input data from the trained model
    :param config:
    :param data: input data to perform predictions on
    :return: numpy array of predictions
    """
    data = torch.tensor(data, dtype=torch.float)
    test_set = TensorDataset(data)
    test_loader = DataLoader(test_set, batch_size=1)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = LSTM_Model(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))

    result = torch.Tensor().to(device)

    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_x = _data[0].to(device)
        pred_x, hidden_predict = model(data_x, hidden_predict)

        cur_pred = torch.squeeze(pred_x, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()

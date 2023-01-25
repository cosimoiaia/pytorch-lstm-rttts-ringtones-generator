import logging
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from config import Config
from data import Data

logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

config = Config()
data = Data(config)


class RTTTLTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead):
        super(RTTTLTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = TransformerEncoder(TransformerEncoderLayer(hidden_size, nhead), num_layers=1)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        transformer_output = self.transformer(embedded.transpose(0, 1))
        out = self.linear(transformer_output.transpose(0, 1))
        return out


vocab_size = len(data.vocab)

model = RTTTLTransformer(vocab_size, config.hidden_size, vocab_size, config.nhead)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

model = model

input_data, output_data = data.get_train_and_valid_data()

# train the model
for epoch in range(1, config.epoch + 1):
    for i, target in zip(input_data, output_data):
        # convert input and target to tensors
        i = torch.tensor(i).view(-1, 1)
        target = torch.tensor(target).view(-1)
        # forward pass
        output = model(i)
        loss = criterion(output.view(-1, vocab_size), target)
        # backward pass and optimization
        model.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch} loss: {loss.item():.4f}')


# define a function to generate a ringtone
def generate_ringtone(title, model):
    # convert title to integers
    prompt = data.encode_text(title)
    prompt = torch.tensor(prompt).view(-1, 1)
    # generate ringtone
    out = model(prompt)
    ringtone = ''
    for i in range(out.shape[0]):
        # get the character with the highest probability
        output_char = data.decode_text([output[i].argmax().item()])
        ringtone += output_char
    return ringtone


test = "takeonme"
result = generate_ringtone(test, model)
print(result)

import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    # class for lstm model
    def __init__(self, params):
        super(LanguageModel, self).__init__()

        # parameters
        if len(params) > 0:
            vocab_size = params["vocab_size"]
            embed_dim = params["embed_dim"]
            hidden_dim = params["hidden_dim"]
            num_layers = params["num_layers"]
            
        # init embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # init lstm layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # forward pass

        # get embedding vectors from input
        x = self.embedding(x)

        # forward lstm and return the last hidden state
        _, (hidden_state, _) = self.lstm(x)
        return hidden_state[-1]


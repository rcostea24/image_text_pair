import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    # mlp class for text
    def __init__(self, params):
        super(LanguageModel, self).__init__()

        # get params
        if len(params) > 0:
            vocab_size = params["vocab_size"]
            embed_dim = params["embed_dim"]
            fc_size = params["fc_size"]
            act = params["act"]
            
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # get activation function and set the first layer
        act_func = getattr(nn, act)
        layers = [
            nn.Linear(fc_size[0], fc_size[1]),
            act_func()
        ]

        # set the rest of the layers
        for id in range(1, len(fc_size)-1):
            layers.append(nn.Linear(fc_size[id], fc_size[id+1]))
            
            if id < len(fc_size):
                layers.append(act_func())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # forward pass

        # get vector for input
        x = self.embedding(x)
        x = x.view([x.shape[0], x.shape[1]*x.shape[2]])

        # pass to mlp
        out = self.mlp(x)
        return out


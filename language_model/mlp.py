import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, params):
        super(LanguageModel, self).__init__()
        if len(params) > 0:
            vocab_size = params["vocab_size"]
            embed_dim = params["embed_dim"]
            fc_size = params["fc_size"]
            act = params["act"]
            
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        act_func = getattr(nn, act)
        layers = [
            nn.Linear(fc_size[0], fc_size[1]),
            act_func()
        ]

        for id in range(1, len(fc_size)-1):
            layers.append(nn.Linear(fc_size[id], fc_size[id+1]))
            
            if id < len(fc_size):
                layers.append(act_func())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view([x.shape[0], x.shape[1]*x.shape[2]])

        out = self.mlp(x)
        return out
    
if __name__ == "__main__":
    model = LanguageModel({
        "vocab_size": 3622, 
        "embed_dim": 32,
        "fc_size": [1024, 512],
        "act": "ReLU"
    }).to("cpu")

    x = torch.randint(3622, size=(64,32)).to("cpu")

    y = model(x)
    print(y.shape)


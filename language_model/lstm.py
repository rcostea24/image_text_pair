import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, params):
        super(LanguageModel, self).__init__()
        if len(params) > 0:
            vocab_size = params["vocab_size"]
            embed_dim = params["embed_dim"]
            hidden_dim = params["hidden_dim"]
            num_layers = params["num_layers"]
            
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden_state, _) = self.lstm(x)
        return hidden_state[-1]
    
if __name__ == "__main__":
    model = LanguageModel({
        "vocab_size": 3622, 
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_layers": 3,
    }).to("cpu")

    x = torch.randint(3622, size=(64,32)).to("cpu")

    y = model(x)
    print(y.shape)


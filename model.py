import torch
import torch.nn as nn
from vision_model.cnn import VisionModel
from language_model.mlp import LanguageModel

class Model(nn.Module):
    def __init__(self, vision_params, language_params, classifier_params):
        super(Model, self).__init__()
        self.vision_model = VisionModel(vision_params)

        if "weights" in vision_params and vision_params["weights"] is not None:
            state_dict = torch.load(vision_params["weights"])
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key in self.vision_model.state_dict():
                    backbone_state_dict[key] = value
            self.vision_model.load_state_dict(backbone_state_dict)
            print("vision weights loaded")
        
        self.language_model = LanguageModel(language_params)

        if "weights" in language_params and language_params["weights"] is not None:
            state_dict = torch.load(language_params["weights"])
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key in self.language_model.state_dict():
                    backbone_state_dict[key] = value
            self.language_model.load_state_dict(backbone_state_dict)
            print("language weights loaded")

        fc_size = classifier_params["fc_size"]
        cls_act = getattr(nn, classifier_params["act"])
        cls_layers = [
            nn.BatchNorm1d(fc_size[0]),
            nn.Linear(
                fc_size[0], 
                fc_size[1]
            ),
            cls_act()
        ]

        for id in range(1, len(fc_size)-1):
            cls_layers.append(nn.Linear(fc_size[id], fc_size[id+1]))
            
            if id < len(fc_size):
                cls_layers.append(cls_act())

        cls_layers.append(nn.Softmax(dim=1))

        self.classifier = nn.Sequential(*cls_layers)

    def forward(self, img_x, txt_x):
        vision_embed = self.vision_model(img_x)
        language_embed = self.language_model(txt_x)

        classifier_input = torch.cat([vision_embed, language_embed], dim=1)
        out = self.classifier(classifier_input)

        return out
import torch
import torchvision.models as models


def get_model_by_name(model_name, pretrained=True):
    model = models.__dict__[model_name](pretrained=pretrained)
    model = torch.nn.Sequential(*list(model.children()))[:-1]
    for param in model.parameters():
        model.requires_grad = False
    return model

from collections import OrderedDict

import torch
import torchvision.models as models


def get_cpu_copy(model):
    return OrderedDict({
        k: v.to("cpu")
        for k, v in model.state_dict().items()
    })


def get_model_by_name(model_name, pretrained=True):
    model = models.__dict__[model_name](pretrained=pretrained)
    model = torch.nn.Sequential(*list(model.children()))[:-1]
    for param in model.parameters():
        model.requires_grad = False
    return model


def save_checkpoint(checkpoint_path, **components):
    checkpoint_dict = dict()
    for name, component in components.items():
        if hasattr(component, "state_dict"):
            checkpoint_dict[name] = component.state_dict()
        else:
            checkpoint_dict[name] = component
    torch.save(checkpoint_dict, checkpoint_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import torchvision.models as models

"""
DVBPR -- PyTorch port
Paper: http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm17.pdf
Original implementation: https://github.com/kang205/DVBPR/blob/master/DVBPR/main.py

Note that we do not consider the GAN element of the paper in this work.
"""


class CNN(nn.Module):
    def __init__(self, hidden_dim=2048, fc_dim=64, weights=None, dropout=0.5):
        super(CNN, self).__init__()
        self.hidden_dim = hidden_dim

        if weights is None:
            # set default network dimensions
            weights = {
                # conv layers: ((c_in, c_out, stride (square)), custom stride)
                'cnn': [([3, 64, 11], [1, 4]),
                        ([64, 256, 5], None),
                        ([256, 256, 3], None),
                        ([256, 256, 3], None),
                        ([256, 256, 3], None)],
                    
                # fc layers: n_in, n_out
                'fc': [[256*22*2, fc_dim],  # original: 256*7*7 -> 4096
                    # [fc_dim, fc_dim],
                    [fc_dim, self.hidden_dim]]
            }

        self.convs = nn.ModuleList([nn.Conv2d(*params, padding_mode='replicate', stride=stride if stride else 1)
                                    for params, stride in weights['cnn']])
        
        self.fcs = nn.ModuleList([nn.Linear(*params) for params in weights['fc']])
        
        self.maxpool2d = nn.MaxPool2d(2)
        self.maxpool_idxs = [True, True, False, False, True]  # CNN layers to maxpool
        self.dropout = nn.Dropout(p=dropout)        
        self.layer_params = weights

    def forward(self, x):
        # reshape input picture
        x = torch.reshape(x, shape=[-1, 3, 224, 224])

        # convolutional layers
        for cnn_layer, apply_maxpool in zip(self.convs, self.maxpool_idxs):
            x = F.relu(cnn_layer(x))
            # notable difference: original TF implementation has "SAME" padding, might be worth trying out
            x = self.maxpool2d(x) if apply_maxpool else x

        # reshape between conv and linear
        x = torch.reshape(x, shape=[-1, self.layer_params['fc'][0][0]])

        # fully connected layers
        for fc_layer in self.fcs:
            x = F.relu(fc_layer(x))
            x = self.dropout(x)

        return x

    def reset_parameters(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)


class DVBPR(nn.Module):
    def __init__(self, n_users, n_items, features, K=2048):
        super().__init__()
        self.cache = None

        # CNN for learned image features
        # tried premade architectures for faster training:
        # squeezenet, cnnf, resnet: 4hrs epoch, no go
        # alexnet: 1hrs epoch, to beat
        # densenets: 18hrs
        # mnasnet: 2:50 hrs
        # self.cnn = CNN(hidden_dim=K)
        model = models.alexnet() #pretrained=True)

        # freeze
        #for param in model.parameters():
        #    param.requires_grad = False

        len_final_layer = model.classifier[-1].weight.shape[1]  # (model.classifier.children())[:-1][1].weight.shape[0]
        model.classifier[-1] = nn.Linear(len_final_layer, K)
        # model.classifier = nn.Linear(len_final_layer, K)
        # model.classifier = model.classifier[:-1] # Full freeze
        self.cnn = model
        # self.cnn = nn.Embedding.from_pretrained(torch.Tensor(features))#, freeze=False)

        # Visual latent preference (theta)
        self.theta_users = nn.Embedding(n_users, K)

        # Latent factors (gamma)
        self.gamma_users = nn.Embedding(n_users, 100)
        self.gamma_items = nn.Embedding(n_items, 100)

        # User bias
        # self.beta_users = nn.Embedding(n_users, 1)

        # Random weight initialization
        self.reset_parameters()

    def forward(self, ui, pimg, nimg, pi, ni):
        """Forward pass of the model.

        Feed forward a given input (batch). Each object is expected
        to be a Tensor.

        Args:
            ui: User index, as a Tensor.
            pi: Positive item index, as a Tensor.
            ni: Negative item index, as a Tensor.

        Returns:
            Network output (scalar) for each input.
        """

        # User
        ui_visual_factors = self.theta_users(ui)  # Visual factors of user u
        # ui_bias = self.beta_users(ui)
        ui_latent_factors = self.gamma_users(ui)  # Latent factors of user u

        # Items
        pi_features = self.cnn(pimg)  # Pos. item visual features
        ni_features = self.cnn(nimg)  # Neg. item visual features

        pi_latent_factors = self.gamma_items(pi)  # Pos. item visual factors
        ni_latent_factors = self.gamma_items(ni)  # Neg. item visual factors

        x_ui = (ui_visual_factors * pi_features).sum(1) + (pi_latent_factors * ui_latent_factors).sum(1)  # + ui_bias.squeeze()
        x_uj = (ui_visual_factors * ni_features).sum(1) + (ni_latent_factors * ui_latent_factors).sum(1)  # + ui_bias.squeeze()

        return x_ui, x_uj

    def recommend_all(self, user, img_list, cache=None, grad_enabled=False):
        with torch.set_grad_enabled(grad_enabled):
            # User
            u_visual_factors = self.theta_users(user)  # Visual factors of user u
            u_bias = self.beta_users.weight
            # Items
            if cache is not None:
                visual_rating_space = cache
            elif self.cache is not None:
                visual_rating_space = self.cache
            else:
                visual_rating_space = self.generate_cache(img_list)

            x_ui = u_bias + (u_visual_factors * visual_rating_space).sum(dim=1).unsqueeze(-1)

            return x_ui

    def reset_parameters(self):
        """ Restart network weights using a Xavier uniform distribution. """
        nn.init.uniform_(self.theta_users.weight)  # Visual factors (theta)
        nn.init.uniform_(self.gamma_users.weight)  # Visual factors (theta)
        nn.init.uniform_(self.gamma_items.weight)  # Visual factors (theta)
        # nn.init.xavier_uniform_(self.beta_users.weight)  # Biases (beta)
        # self.cnn.reset_parameters() # CNN

    def generate_cache(self, img_list, grad_enabled=False, device='cpu'):
        cache = []
        with torch.set_grad_enabled(grad_enabled):
            for img in tqdm(img_list):
                img = img.to(device).unsqueeze(0)
                cache.append(self.cnn(img))
            self.cache = torch.stack(cache)
            return cache

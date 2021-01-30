import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DVBPR -- PyTorch port
Author: Patricio Cerda Mardini <pcerdam@uc.cl>
Paper: http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm17.pdf
Original implementation: https://github.com/kang205/DVBPR/blob/master/DVBPR/main.py

Note that we do not consider the GAN element of the paper in this work.

To download original dataset:
git clone https://github.com/kang205/DVBPR
cd ./content/DVBPR
./download_dataset.sh
"""


class CNN(nn.Module):
    def __init__(self, weights, dropout=0.5):
        super(CNN, self).__init__()
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
        x = torch.reshape(x, shape=[-1,self.layer_params['fc'][0][0]])

        # fully connected layers
        for fc_layer in self.fcs:
            x = F.relu(fc_layer(x))
            x = self.dropout(x)

        return x


if __name__ == '__main__':
    
    load_amazon_dataset = False
    if load_amazon_dataset:
        dataset_name = './AmazonFashion6ImgPartitioned.npy'
        dataset = np.load(dataset_name, allow_pickle=True, encoding='bytes')

        # dict, dict, dict, dict, int, int
        [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
        print(f'Loaded Amazon Fashion dataset, with {usernum} users and {itemnum} items.')

    # set hyperparameters
    K = 100               # latent dimensionality
    lambda1 = 0.001       # weight decay
    lambda2 = 1.0         # regularizer for theta_u
    learning_rate = 1e-4
    training_epoch = 20
    batch_size = 128
    dropout = 0.5
    numldprocess= 4       # multi-threading for loading images

    # set network dimensions
    weights = {
        # conv layers: ((c_in, c_out, stride (square)), custom stride)
        'cnn': [([3, 64, 11], [1, 4]),
                ([64, 256, 5], None),
                ([256, 256, 3], None),
                ([256, 256, 3], None), 
                ([256, 256, 3], None)],
            
        # fc layers: n_in, n_out
        'fc': [[256*22*2, 4096],  # original: 256*7*7 -> 4096
            [4096, 4096],
            [4096, K]]
    }

    # up to here, covers until line 110 of original implementation
    cnn = CNN(weights)
    print(cnn(torch.randn(1, 3, 224, 224)))

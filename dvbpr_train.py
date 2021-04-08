import os
import random
from PIL import Image

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from datasets import UserModeImgDataset, UserModeDataset, UserModeFeatDataset
from models import DVBPR
from trainers import ImgTrainer
from trainers.losses import bpr_loss
from utils.data import extract_embedding

if __name__ == '__main__':
    # Parameters
    RNG_SEED = 0
    BASE_PATH = '/home/pcerdam/VisualRecSys-Tutorial-IUI2021/'
    TRAINING_PATH = os.path.join(BASE_PATH, "data", "naive-user-train.csv")
    EMBEDDING_PATH = os.path.join(BASE_PATH, "data", "embedding-resnet50.npy")
    VALIDATION_PATH = os.path.join(BASE_PATH, "data", "naive-user-validation.csv")
    IMAGES_PATH = os.path.join('/mnt/data2/wikimedia/mini-images-224-224-v2')
    CHECKPOINTS_DIR = os.path.join(BASE_PATH, "checkpoints")
    version = f"DVBPR_wikimedia_resnetEmbTable"
    USE_GPU = True # False #
    version = 'DVBPR_wikimediaAlexNet_notPretrained_100_wLatent'

    # Parameters (training)
    SETTINGS = {
        "dataloader:batch_size": 128, # 256,  #  512, # 64,  # 64,  # 24,  # 42_000,128,  # x
        "dataloader:num_workers": 4, # os.cpu_count(),  # 1,  #
        "prev_checkpoint": False, # 'DVBPR_wikimediaAlexNetBig204_5epochs',
        "model:dim_visual": 100, #2048,
        "optimizer:lr": 0.001,
        "optimizer:weight_decay": 0.0001,
        "scheduler:factor": 0.6,
        "scheduler:patience": 2,
        "train:max_epochs": 5,  # 1, # 5,  # 150,
        "train:max_lrs": 5,
        "train:non_blocking": True,
        "train:train_per_valid_times": 1  # 0

    }

    # ================================================

    # Freezing RNG seed if needed
    if RNG_SEED is not None:
        print(f"\nUsing random seed...")
        random.seed(RNG_SEED)
        torch.manual_seed(RNG_SEED)
        np.random.seed(RNG_SEED)

    # Load embedding from file
    print(f"\nLoading embedding from file... ({EMBEDDING_PATH})")
    embedding = np.load(EMBEDDING_PATH, allow_pickle=True)

    # Extract features and "id2index" mapping
    print("\nExtracting data into variables...")
    embedding, id2index, index2fn = extract_embedding(embedding, verbose=True)
    print(f">> Features shape: {embedding.shape}")

    # DataLoaders initialization
    print("\nInitialize DataLoaders")
    # Training DataLoader
    train_dataset = UserModeImgDataset( # UserModeDataset(  #
        csv_file=TRAINING_PATH,
        img_path=IMAGES_PATH,
        id2index=id2index,
        index2fn=index2fn
    )
    print(f">> Training dataset: {len(train_dataset)}")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        #Subset(train_dataset, list(range(10000))), #  subset for faster tests
        batch_size=SETTINGS["dataloader:batch_size"],
        num_workers=SETTINGS["dataloader:num_workers"],
        shuffle=True,
        pin_memory=True,
    )
    print(f">> Training dataloader: {len(train_dataloader)}")
    # Validation DataLoader
    valid_dataset = UserModeImgDataset( # UserModeDataset(  #
        csv_file=VALIDATION_PATH,
        img_path=IMAGES_PATH,
        id2index=id2index,
        index2fn=index2fn
    )
    print(f">> Validation dataset: {len(valid_dataset)}")
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(
        #Subset(valid_dataset, list(range(10000))),  #  subset for faster tests
        valid_dataset,
        batch_size=SETTINGS["dataloader:batch_size"],
        num_workers=SETTINGS["dataloader:num_workers"],
        shuffle=True,
        pin_memory=True,
    )
    print(f">> Validation dataloader: {len(valid_dataloader)}")
    # Model initialization
    print("\nInitialize model")
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")
    if torch.cuda.is_available() != USE_GPU:
        print((f"\nNotice: Not using GPU - "
               f"Cuda available ({torch.cuda.is_available()}) "
               f"does not match USE_GPU ({USE_GPU})"
               ))
    N_USERS = len(set(train_dataset.ui))
    N_ITEMS = len(embedding)
    print(f">> N_USERS = {N_USERS} | N_ITEMS = {N_ITEMS}")
    print(torch.Tensor(embedding).shape)
    model = DVBPR(
        N_USERS,  # Number of users and items
        N_ITEMS,
        embedding,  # experiments for debugging
        SETTINGS["model:dim_visual"],  # Size of visual spaces
    ).to(device)

    print(model)

    # Training setup
    print("\nSetting up training")
    optimizer = optim.Adam(
        model.parameters(),
        lr=SETTINGS["optimizer:lr"],
        weight_decay=SETTINGS["optimizer:weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss(reduction="sum")  # bpr_loss  # # # nn.MarginRankingLoss(reduction="mean")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SETTINGS["scheduler:factor"],
        patience=SETTINGS["scheduler:patience"], verbose=True,
    )

    # ================================================

    # Training
    trainer = ImgTrainer(
        model, device, criterion, optimizer, scheduler,
        checkpoint_dir=CHECKPOINTS_DIR,
        version=version,
    )
    best_model, best_acc, best_loss, best_epoch = trainer.run(
        SETTINGS["train:max_epochs"], SETTINGS["train:max_lrs"],
        {"train": train_dataloader, "validation": valid_dataloader},
        train_valid_loops=SETTINGS["train:train_per_valid_times"],
        use_checkpoint=SETTINGS["prev_checkpoint"]
    )





from copy import deepcopy
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from torch import tensor
import numpy as np
from tqdm import tqdm

from utils.logger import Log


def generate_collate_fn(max_profile_size, pad_token=0):
    def pad_profile(profile, max_size):
        result = np.full((max_size,), pad_token)
        result[:len(profile)] = profile
        return result

    def collate_fn(batch):
        users, profiles, pos, neg = zip(*batch)
        users, pos, neg = torch.tensor(users), torch.tensor(pos), torch.tensor(neg)
        max_size = max(len(p) for p in profiles)
        max_size = min(max_profile_size, max_size)
        profiles = [pad_profile(profile, max_size) for profile in profiles]
        profiles = torch.tensor(profiles)
        return users, profiles, pos, neg

    return collate_fn


def get_device(device=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    elif isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    else:
        device = torch.device(device)
    return device


class ACFTrainer():
    """
    Handles training process
    """
    def __init__(self, model, datasets, loss, optimizer, run_name, batch_size=100, device=None,
                 test_size=10, max_profile_size=9, pad_token=0):
        """
        Parameters
        ----------
        model: initialized UserNet
        dataset: initialized MovieLens
        loss: one of the warp functions
        optimizer: torch.optim
        run_name: directory to save results
        batch_size: number of samples to process for one update
        device: gpu or cpu
        test_size: number of tail items for each user to leave for test
        """
        self.best_loss = np.inf
        self.loss = loss
        self.model = model
        # self.dataset = dataset
        self.optimizer = optimizer
        self.pad_token = pad_token
        self.batch_size = batch_size
        self.logger = Log(run_name)
        self.device = get_device(device)
        self.model = self.model.to(self.device)
        self.train, self.test = datasets
        self.all_items = self.preprocess_inputs(self.train.items, to_tensor=True)

        self.test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=True,
                                      collate_fn=generate_collate_fn(max_profile_size), num_workers=1)
        self.train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True,
                                       collate_fn=generate_collate_fn(max_profile_size), num_workers=8)

    @property
    def state(self):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state

    def fit(self, num_epochs, k=10):
        num_train_batches = len(self.train) / self.batch_size
        num_test_batches = len(self.test) / self.batch_size
        for epoch in tqdm(range(num_epochs)):
            for phase in ['train', 'val']:
                self.logger.epoch(epoch, phase)
                self.model.train(phase == 'train')
                loss = 0
                cur_step = 0
                if phase == 'train':
                    t = tqdm(self.train_loader)
                    for batch in t:
                        self.optimizer.zero_grad()
                        cur_loss  = self.training_step(batch)
                        self.optimizer.step()
                        loss += cur_loss
                        cur_step += 1
                        avg_loss = loss / cur_step

                        t.set_description(f"Average Loss {avg_loss:.4f}")
                        t.refresh()

                    loss /= num_train_batches
                    self.logger.metrics(loss, 0, epoch, phase)
                else:
                    with torch.no_grad():
                        for batch in tqdm(self.test_loader):
                            cur_loss = self.validation_step(batch)
                            loss += cur_loss
                        loss /= num_test_batches
                        # self.logger.metrics(loss, self.score(k=k), epoch, phase)
                        self.logger.metrics(loss, 0.0, epoch, phase)

                        if loss < self.best_loss:
                            self.best_loss = loss
                            self.logger.save(self.state, epoch)

    def get_profile_mask(self, profile_ids):
        return (profile_ids != self.pad_token).to(self.device)

    def get_predictions(self, user, items):
        item_embeddings = self.model.item_embedding(items)
        prediction = self.model.score(user, item_embeddings)
        return prediction

    def predict(self, user_id, item_ids):
        user = self.get_user_embedding(user_id)
        pred = self.get_predictions(user, item_ids)
        return pred

    def training_step(self, batch):
        user_id, profile_ids, pos, neg = self.preprocess_inputs(*batch)
        profile_mask = self.get_profile_mask(profile_ids)
        pos_pred, neg_pred = self.model(user_id, profile_ids, pos, neg, profile_mask)

        loss = self.loss(pos_pred, neg_pred)
        loss.backward()
        return loss.item()

    def validation_step(self, batch):
        user_id, profile_ids, pos, neg = self.preprocess_inputs(*batch)
        profile_mask = self.get_profile_mask(profile_ids)
        pos_pred, neg_pred = self.model(user_id, profile_ids, pos, neg, profile_mask)

        loss = self.loss(pos_pred, neg_pred)
        return loss.item()

    def preprocess_inputs(self, *inputs, to_tensor=False):
        if to_tensor:
            inputs = tuple(torch.tensor(input_) for input_ in inputs)

        inputs = tuple(input_.long() for input_ in inputs)
        inputs = tuple(input_.to(self.device) for input_ in inputs)
        return inputs

import os
import shutil
import time

import torch
from torch.utils.tensorboard import SummaryWriter


class Log():
    """
    Handles logging for a train process.
    It prints messages, writes them to log file, writes metrics for tensorboard and saves model.
    Everything is saved to runs/<model_name> directory.
    """
    def __init__(self, model_name):
        if not os.path.exists('runs'):
            os.mkdir('runs')

        path = os.path.join('runs', model_name)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)

        self.model_name = model_name
        self.path = path
        self.writer = SummaryWriter(path)

    def log(self, text):
        """
        Write <text> both to log and stdout
        """
        print(text)
        f = open(f"{self.path}/log.txt", "a+")
        f.write(text + '\n')
        f.close()

    def epoch(self, n, phase):
        start = time.strftime("%H:%M:%S")
        if phase == 'train':
            self.log('\n')
        self.log(f"Starting epoch: {n} | phase: {phase} | ⏰: {start}")

    def metrics(self, loss, accuracy, epoch, phase, digits=6):
        self.log(f"Loss: {round(loss, digits)}, Accuracy: {round(accuracy, digits)}")

        self.writer.add_scalar(f'Loss/{phase}', loss, epoch)
        self.writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)

    def save(self, state, epoch):
        self.log("******** New optimal found, saving state ********")
        state['epoch'] = epoch
        torch.save(state, f"{self.path}/{self.model_name}_e{epoch}.pt")
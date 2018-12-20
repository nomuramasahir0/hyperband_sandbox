import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.mnist_loader import mnist_data_loader
from torch.optim.sgd import SGD
import os
import numpy as np
import uuid


class Network(nn.Module):

    def __init__(self, hparam):
        self.name = 'MLPWithMNIST'
        super(Network, self).__init__()
        fc1_unit = hparam['fc1_unit']
        # self.fc1 = nn.Linear(28 * 28 * 1, fc1_unit)
        self.fc1 = nn.Linear(784, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc1_unit)
        self.fc3 = nn.Linear(fc1_unit, 10)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


class MLPWithMNIST:

    def __init__(self, hparams, ckpt_name, homedir, separate_history):
        self.hparams = hparams
        # batch size
        self.batch_size = 64
        # loader
        self.loader_train, self.loader_valid, self.loader_test = mnist_data_loader(self.batch_size)
        # model
        self.model = Network(hparams)

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # initial learning rate
        self.lr = hparams['lr']
        # momentum coef
        self.momentum = hparams['momentum']
        # optimizer
        self.optimizer = SGD(self.model.parameters(),
                             lr=self.lr,
                             momentum=self.momentum,
                             nesterov=True)
        # device
        self.num_gpu = 0
        self.device = torch.device("cuda" if self.num_gpu > 0 else "cpu")

        self.ckpt_dir = homedir + "ckpt"
        self.ckpt_name = ckpt_name

        # epoch
        self.epoch = 0

        # history
        self.separate_history = separate_history

        try:
            ckpt = self._load_checkpoint(self.ckpt_name)
            self.model.load_state_dict(ckpt['state_dict'])
            self.epoch = ckpt['current_epoch']
        except FileNotFoundError:
            pass

    def evaluate(self, num_iter):

        min_val_loss = np.inf
        diff_epoch = num_iter - self.epoch
        for epoch in range(diff_epoch):
            self._train_one_epoch()
            self.epoch += 1
            print("self.epoch:{}".format(self.epoch))
            val_loss = self._validate_one_epoch()
            self.separate_history[self.ckpt_name].append((self.hparams, val_loss))
            print("val_loss:{}".format(val_loss))
            if val_loss < min_val_loss:
                min_val_loss = val_loss
        state = {
            'state_dict': self.model.state_dict(),
            'min_val_loss': min_val_loss,
            'current_epoch': self.epoch
        }
        self._save_checkpoint(state, self.ckpt_name)
        return min_val_loss

    def _train_one_epoch(self):
        self.model.train()
        for data, targets in self.loader_train:
            self.model.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def _validate_one_epoch(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, targets in self.loader_valid:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == targets).sum().item()

        data_num = len(self.loader_valid.dataset)
        val_loss = (1 - correct / data_num) * 100
        return val_loss

    def _save_checkpoint(self, state, name):
        filename = name + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

    def _load_checkpoint(self, name):
        filename = name + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        return ckpt

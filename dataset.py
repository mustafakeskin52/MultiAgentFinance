"""
Ugur Gudelek
dataset
ugurgudelek
06-Mar-18
finance-cnn
"""
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import config

import torch
from torch import nn

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from tqdm import trange, tqdm
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import os
import warnings
from sklearn import preprocessing
from collections import defaultdict
from scipy import spatial

import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
from torch import FloatTensor,LongTensor
from torch.autograd import Variable

def dataset_by_name(name):
    if name == "MNISTDataset":
        return MNISTDataset
    if name == "DATASETSequence":
        return DATASETSequence
    if name == "IndicatorDataset":
        return IndicatorDataset
class DATASETSequence():
    def __init__(self,config):
        length = 100
        alldata = np.array([i for i in range(length)])

        #alldata = np.array([100]*100)

        raw_dataset_y = pd.DataFrame(alldata[:])
        raw_dataset_x = pd.DataFrame(alldata[:])

        train_len = int(raw_dataset_x.shape[0] * config.TRAIN_VALID_RATIO)

        self.raw_dataset_x_training = raw_dataset_x.iloc[:train_len]
        self.raw_dataset_x_validation = raw_dataset_x.iloc[train_len:]
        self.raw_dataset_y_training = raw_dataset_y.iloc[:train_len]
        self.raw_dataset_y_validation = raw_dataset_y.iloc[train_len:]

        self.train_dataset = InnerIndicatorDataset(datasetX=self.raw_dataset_x_training,
                                                   datasetY=self.raw_dataset_y_training, seq_len=config.SEQ_LEN,
                                                   problem_type=config.LABEL_TYPE)
        self.valid_dataset = InnerIndicatorDataset(datasetX=self.raw_dataset_x_validation,
                                                   datasetY=self.raw_dataset_y_validation,seq_len=config.SEQ_LEN,
                                                   problem_type=config.LABEL_TYPE)

    def random_sample(self, n):
        perm = np.random.randint(0, self.train_dataset.__len__(), size=n)
        xs, ys = self.train_dataset.get_all_data()

        return torch.Tensor(xs[perm]), ys[perm]
class MNISTDataset():
    def __init__(self, config):

        self.train_dataset = datasets.MNIST('../dataset', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ])
                                            )
        self.valid_dataset = datasets.MNIST('../dataset', train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ])
                                            )
        # todo: check for transformation later


    def random_sample(self, n):
        perm = np.random.randint(0, self.train_dataset.__len__(), size=n)
        data = self.train_dataset.train_data.numpy()[perm]
        labels = self.train_dataset.train_labels.numpy()[perm]

        return torch.Tensor(data).unsqueeze(dim=1), labels

class InnerIndicatorDataset(torch.utils.data.Dataset):
    """
    Args:
        dataset(pd.DataFrame):
    """

    def __init__(self, datasetX,datasetY, seq_len, problem_type):
        self.X = datasetX
        self.y = datasetY

        if problem_type=='classification':
            # turn categorical to one hot encoding
            self.y = pd.get_dummies(self.y)

        self.feature_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]
        self.data_dim = self.X.shape[0]
        self.seq_len = seq_len

        #self._X = self.X.values.reshape(-1, self.feature_dim, self.seq_len)
        #self._y = self.y.values.reshape(-1, self.output_dim, self.seq_len)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.X.shape[0] - self.seq_len

    def __getitem__(self, ix):
        X = self.X.iloc[ix: ix + self.seq_len, :]
        y = self.y.iloc[ix + self.seq_len, :]

        # change type to numpy
        X = X.values.astype(float)
        y = y.values.astype(float)

        X = np.expand_dims(X, axis=0)
        y = np.expand_dims(y, axis=0)

        return X, y

    def get_all_data(self, transforms=None):

        xs, ys = self.__getitem__(0)

        for ix in range(1,self.__len__()):
            X, y = self.__getitem__(ix)
            xs = np.append(xs, X, axis=0)
            ys = np.append(ys, y, axis=0)

        # tranform
        # example:
        # transforms=[FloatTensor, Variable])
        # xs = Variable(FloatTensor(xs))
        if transforms is not None:
            for transform in transforms:
                xs, ys = transform(xs), transform(ys)
        return xs, ys#ys.unsqueeze_(-1).resize_(84,3)

    def _reshape(self, data):
        # (in_channels, width, height)
        return data.reshape((1, data.shape[0], data.shape[1]))

    def get_sample(self):
        ix = np.random.randint(low=0, high=self.__len__())
        return ix, self.__getitem__(ix=ix)

class IndicatorDataset():
    """
    """

    def __init__(self,config):

        self.input_path_x = config.INPUT_PATH_X
        self.input_path_y = config.INPUT_PATH_Y
        self.train_valid_ratio = config.TRAIN_VALID_RATIO
        self.seq_len = config.SEQ_LEN
        self.label_type = config.LABEL_TYPE

        raw_dataset_x = pd.read_csv(self.input_path_x)
        raw_dataset_y = pd.read_csv(self.input_path_y)
        train_len = int(raw_dataset_x.shape[0] * self.train_valid_ratio)
        self.raw_dataset_x_training = raw_dataset_x.iloc[:train_len, :]
        self.raw_dataset_x_validation = raw_dataset_x.iloc[train_len:, :]
        self.raw_dataset_y_training = raw_dataset_y.iloc[:train_len, :]
        self.raw_dataset_y_validation = raw_dataset_y.iloc[train_len:, :]

        # if save_dataset:
        #     self.preprocessed_train_dataset.to_csv(
        #         os.path.join('/'.join(input_path.split('/')[:-1]), 'train_preprocessed_indicator_dataset.csv'),
        #         index=False)
        #     self.preprocessed_valid_dataset.to_csv(
        #         os.path.join('/'.join(input_path.split('/')[:-1]), 'valid_preprocessed_indicator_dataset.csv'),
        #         index=False)

        self.train_dataset = InnerIndicatorDataset(datasetX=self.raw_dataset_x_training,datasetY=self.raw_dataset_y_training,seq_len=self.seq_len, problem_type=self.label_type)
        self.valid_dataset = InnerIndicatorDataset(datasetX=self.raw_dataset_x_validation,datasetY = self.raw_dataset_y_validation, seq_len=self.seq_len, problem_type=self.label_type)

if __name__ == "__main__":

    dataset = DATASETSequence(config.ConfigLSTM(),100,0.8,3,"regression")
    xs,ys = dataset.train_dataset.get_all_data()
    #
    #print(dataset.train_dataset)
    train_xs, train_ys = dataset.train_dataset.get_all_data()
    valid_xs, valid_ys = dataset.valid_dataset.get_all_data()

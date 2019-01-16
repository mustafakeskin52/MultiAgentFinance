import config
import torch
from torch import nn

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from tqdm import trange, tqdm
import pandas as pd
import numpy as np

import time

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix

import os
import torchvision

import model
import dataset
from Experiment import Experiment

if __name__ == "__main__":

    """
    1. Implement Dataset Class
    2. Implement Model Class
    3. Configure configClass
    4. Pass config to experiment
    5. Run
    """
    config = config.ConfigMLP()
    #
    #dataset = dataset.CNN1DDataSet(seq_len=20)
    #dataset = dataset.MNISTDataset(config)

    #config = config.ConfigLSTM()
    #dataset = dataset.SequenceLearningOneToOne()
    # model = model.LSTM(input_size=10, seq_length=1, num_layers=1,
    #                    out_size=10, hidden_size=10, batch_size=1, device=config.DEVICE)
    dataX = [[(i%4)/4,(i%4)/4.0] for i in range(0,1000)]
    dataY = []
    for i in range(len(dataX)):
        if i%4 == 3:
            dataY.append(0)
        elif i%4 == 2:
            dataY.append(1)
        elif i%4 == 1:
            dataY.append(2)
        else :
            dataY.append(3)
    print(np.asarray(dataX))
    print(np.asarray(dataY))
    dataset = dataset.MLPToyDataset(dataX = np.asarray(dataX),dataY = np.asarray(dataY))
    #model = model.LSTM(input_size=config.INPUT_SIZE, seq_length=config.SEQ_LEN, num_layers=2,
     #                      out_size=config.OUTPUT_SIZE, hidden_size=5, batch_size=config.TRAIN_BATCH_SIZE,
     #                      device=config.DEVICE)
    #
    model = model.MLP(input_size=2,output_size=4)
    #

    experiment = Experiment(config=config, model=model, dataset=dataset)
    experiment.run()


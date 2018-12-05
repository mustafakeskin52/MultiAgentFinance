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
    config = config.ConfigCNN()
    #
    #dataset = dataset.CNN1DDataSet(seq_len=20)
    #dataset = dataset.MNISTDataset(config)

    #config = config.ConfigLSTM()
    # dataset = dataset.SequenceLearningOneToOne()
    # model = model.LSTM(input_size=10, seq_length=1, num_layers=1,
    #                    out_size=10, hidden_size=10, batch_size=1, device=config.DEVICE)
    config.save()
    dataset = dataset.FinancialDataSet(seq_len=10)
    # model = model.LSTM(input_size=config.INPUT_SIZE, seq_length=config.SEQ_LEN, num_layers=2,
    #                       out_size=config.OUTPUT_SIZE, hidden_size=5, batch_size=config.TRAIN_BATCH_SIZE,
    #                       device=config.DEVICE)
    #
    model = model.CNN(config)
    #

    #experiment = Experiment(config=config, model=model, dataset=dataset)
    #experiment.run()


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


def convert_to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
class Experiment:

    def __init__(self, config, model ,dataset):
        self.config = config
        self.model = model.to(self.config.DEVICE)
        self.dataset = dataset

        self.load()
        # self.save()


    def load(self):

        #self.dataset = dataset_by_name(self.config.DATASET_NAME)(config=self.config)  # MNISTDataset, IndicatorDataset, LoadDataset

        self.train_dataloader = DataLoader(self.dataset.train_dataset,
                                      batch_size=self.config.TRAIN_BATCH_SIZE,
                                      shuffle=self.config.TRAIN_SHUFFLE,
                                      drop_last=True)
        self.valid_dataloader = DataLoader(self.dataset.valid_dataset,
                                      batch_size=self.config.VALID_BATCH_SIZE,
                                      shuffle=self.config.VALID_SHUFFLE,
                                      drop_last=True)
        #MODEL = class_by_name(self.config.MODEL_NAME)  # CNN, LSTM
        #self.model = MODEL(config=self.config).to(self.config.DEVICE)

        self.writer = SummaryWriter(log_dir=os.path.join(self.config.EXPERIMENT_DIR, 'summary'))

    def save(self):
        self.config.save()
        self.model.to_onnx(directory=self.config.EXPERIMENT_DIR)
        self.model.to_txt(directory=self.config.EXPERIMENT_DIR)
    def predict_lstm(self,X,categoricalN):
        # seq_len, dataset_len,input_size
        X = np.asarray(X)
        X = convert_to_categorical(X, categoricalN)
        X = np.expand_dims(X,axis=0)

        print("ShapeOf",X.shape)
        predicted_label = self.model.predict(torch.FloatTensor(X)).cpu().detach()
        return predicted_label

    def predict_cnn(self, X):
        # seq_len, dataset_len,input_size
        X = np.asarray(X)
        X = np.expand_dims(X,axis=0)
        X = np.expand_dims(X,axis=0)
        predicted_label = self.model.predict(torch.FloatTensor(X)).cpu().detach()
        return predicted_label
    def run_epoch(self, epoch):

        self.model.init_hidden()
        for step, (X, y) in enumerate(self.train_dataloader):
            # Fit the model
            self.model.fit(X, y)
            training_loss = self.model.training_loss
        #TODO:The code block is implemented for testing.After all mission is completed,the parameters should be generalized
        score = 0.
        #possibleStates = 2
        #agentNumbers = 3
        states = [-1,1]
        #TODO:Inputdatas is a array keeping statistical information such as counts or histogram and it will be implemented inside statisticalDatas method
        #inputdatas = np.zeros((possibleStates ^ agentNumbers, agentNumbers + 2))
        #labelDatas = np.zeros((np.power(2, agentNumbers), agentNumbers + 2))
        self.model.init_hidden()
        for step, (X, y) in enumerate(self.valid_dataloader):

            # Validate validation set
            self.model.validate(X, y)  # todo: current build call .validate when .score is used!
            #labelDatas = labelDatas + self.model.statisticalDatas(X,y,agentNumbers,labelDatas,inputdatas) # The method is called due to show distrubition of the input or output
            # Score
            score += self.model.score(X, y)
            validation_loss = self.model.validation_loss

        #print("labelDatas", labelDatas)
        score = score/self.valid_dataloader.__len__()
        # Predict
        #predicted_labels = self.model.predict(X_sample).cpu().detach()
        # predicted_labels = prediction_logprob

        # Log
        print("========================================")
        print("Training Loss: {}".format(training_loss))
        print("Validation Loss: {}".format(validation_loss))
        print("Score: {}".format(score))
        #print("X_sampleshape",X_sample.shape)
        # print('Actual label:', self.dataset.sequentialClass[y_sample])
        # print('Sample label:',self.dataset.sequentialClass[torch.argmax(X_sample, 2)])
        # print('Predicted label:', self.dataset.sequentialClass[predicted_labels])
        print("========================================")

        # Write losses to the tensorboard
        self.writer.add_scalar('training_loss', training_loss, epoch)
        self.writer.add_scalar('validation_loss', validation_loss, epoch)

        # # Write random image to the summary writer.
        # image_grid = torchvision.utils.make_grid(X_sample, normalize=True, scale_each=True)
        # self.writer.add_image(tag="RandomSample y-{} yhat{}".format(
        #     '.'.join(map(str, y_sample)), '.'.join(map(str, predicted_labels))),
        #                       img_tensor=image_grid, global_step=epoch)


        # Write PR Curve to the summary writer.
        self.writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), epoch)

        # for name, param in model.named_parameters():
        #     print(name)
        #     print(param)
        #     model.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch, bins=100)
        # x = dict(model.named_parameters())['conv1.weight'].clone().cpu().data.numpy()
        # kernel1= x[0,0]
        # plt.imshow(kernel1)
        # plt.show()
        # needs tensorboard 0.4RC or later

        return training_loss, validation_loss

    def run(self):
        epoch = 0
        with tqdm(total=self.config.EPOCH_SIZE) as pbar:

            for epoch in range(self.config.EPOCH_SIZE):
                tloss,vloss = self.run_epoch(epoch=epoch)
                pbar.set_description("{}||||{}".format(tloss, vloss))
                pbar.update(1)

        self.writer.export_scalars_to_json(self.config.EXPERIMENT_DIR+'.json')
import os
import torch
import time

class ConfigLSTM:
    def __init__(self):
        """
        """
        self.RANDOM_SEED = 42
        self.MODEL_NAME = 'LSTM'
        self.EPOCH_SIZE = 100

        self.SEQ_LEN = 3
        self.INPUT_SIZE = 1
        self.OUTPUT_SIZE = 1
        self.NUM_LAYERS = 4
        self.HIDDEN_SIZE = 40

        # self.LABEL_WINDOW = 7
        self.LABEL_TYPE = 'regression'

        self.TRAIN_VALID_RATIO = 0.80
        self.TRAIN_BATCH_SIZE = 10
        self.VALID_BATCH_SIZE = 10
        self.TRAIN_SHUFFLE = False
        self.VALID_SHUFFLE = False

        self.DATASET_NAME = 'DATASETSequence'
        self.INPUT_PATH_X = 'xtrain.csv'
        self.INPUT_PATH_Y = 'ytrain.csv'
        self.EXPERIMENT_DIR = '../experiment/xlf_' + str(int(time.time()))

        self.USE_CUDA = torch.cuda.is_available()
        if self.USE_CUDA:
            if torch.cuda.get_device_name(0) == 'GeForce GT 650M':
                self.USE_CUDA = False
                print('USE_CUDA is set to False because this GPU is too old.')

        if self.USE_CUDA:
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'

        print('CUDA AVAILABLE:{}'.format(self.USE_CUDA))

    def __str__(self):
        string = ''
        for attr_key, attr_val in self.__dict__.items():
            string += attr_key
            string += '='
            string += str(attr_val)
            string += '\n'
        return string

    def save(self):
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        path = os.path.join(self.EXPERIMENT_DIR, 'config.ini')

        with open(path, 'w') as file:
            file.write(self.__str__())

class ConfigCNN:
    """
    """

    def __init__(self):
        """
        """
        # Aux params
        self.RANDOM_SEED = 42

        # Model params
        self.MODEL_NAME = 'CNN'
        self.INPUT_SIZE = 1
        self.OUTPUT_SIZE = 10

        # Dataloader params
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False
        self.TRAIN_BATCH_SIZE = 100
        self.VALID_BATCH_SIZE = 100

        # Dataset params
        self.TRAIN_VALID_RATIO = 0.90
        self.DATASET_NAME = 'MNISTDataset'

        # Experiment params
        self.EPOCH_SIZE = 100
        self.EXPERIMENT_DIR = '../experiment/{}/{}'.format(self.DATASET_NAME ,str(int(time.time())))

        # Device params
        self.USE_CUDA = torch.cuda.is_available()
        if self.USE_CUDA:
            if torch.cuda.get_device_name(0) == 'GeForce GT 650M':
                self.USE_CUDA = False
                print('USE_CUDA is set to False because this GPU is too old.')

        if self.USE_CUDA:
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'

        print('CUDA AVAILABLE:{}'.format(self.USE_CUDA))

    def __str__(self):
        string = ''
        for attr_key, attr_val in self.__dict__.items():
            string += attr_key
            string += '='
            string += str(attr_val)
            string += '\n'
        return string

    def save(self):
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        path = os.path.join(self.EXPERIMENT_DIR, 'config.ini')

        with open(path, 'w') as file:
            file.write(self.__str__())
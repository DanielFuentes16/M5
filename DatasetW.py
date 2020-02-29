import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
import os
from PIL import Image
import glob

class DatasetW(Dataset):
    """
    A customized data loader.
    """

    def __init__(self, train):
        'Initialization'
        if train:
            pathTrain = "train"
        else:
            pathTrain = "test"

        list_IDs = glob.glob("datasets/MIT_split/"+pathTrain +"/*/*.jpg")
        labels = dict()
        for id in list_IDs:
            labels[id] = id.split("/")[3]

        self.labels = labels
        self.list_IDs = list_IDs
        self.train = train

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        if self.train:
            pathTrain = "train"
        else:
            pathTrain = "test"

        # Load data and get label
        X = torch.load(ID)
        y = self.labels[ID]

        return X, y
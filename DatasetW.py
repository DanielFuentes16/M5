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
        y = self.labels[ID+ '.pt']

        return X, y

class TESNamesDataset(Dataset):
    def __init__(self, data_root, charset, length):
        self.data_root = data_root
        self.charset = charset + '\0'
        self.length = length
        self.samples = []
        self.char_codec = LabelEncoder()
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, pixels = self.samples[idx]
        return self.one_hot_sample(name), pixels

    def _init_dataset(self):
        names = set()
        # self.samples = []
        for animal in os.listdir(self.data_root):
            animal_filepath = os.path.join(self.data_root, animal)
            names.add(animal)
            for img_name in os.listdir(animal_filepath):
                img_path = os.path.join(animal_filepath, img_name)
                im = cv2.imread(img_path)
                if len(animal) < self.length:
                    animal += '\0' * (self.length - len(animal))
                else:
                    animal = animal[:self.length - 1] + '\0'
                self.samples.append((animal, im))

        self.char_codec.fit(list(self.charset))

    def to_one_hot(self, codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_))[value_idxs]

    def one_hot_sample(self, name):
        t_name = self.to_one_hot(self.char_codec, list(name))
        return t_name

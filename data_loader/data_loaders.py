from torchvision import datasets, transforms
from base import BaseDataLoader

from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

######################################################
#                       Cervical                     #
######################################################

class CervicalDataset(Dataset):
    '''
    dataloader에서 쓰일 custom dataset
    '''
    def __init__(self, data_dir, label_dir,  transform=None ):
        self.label_dir = pd.read_csv(label_dir)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        img_path = self.data_dir + "{}_01.jpg".format(self.label_dir.iloc[idx, 1])
        image = Image.open(img_path)
        # image = np.array(image)
        
        if self.transform:
            image = self.transform(image)
            
        image = torch.FloatTensor(image)
        label = self.label_dir.iloc[idx, -1]

        return image, label

from torchvision import datasets, transforms
# from base import BaseDataLoader


class CervicalDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, label_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
        self.data_dir = data_dir
        self.label_dir = label_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=train_transform)
        self.dataset = dataset = CervicalDataset(self.data_dir, self.label_dir, transform = train_transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

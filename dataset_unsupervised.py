import os
import numpy as np
from os.path import join
from scipy.io import loadmat
import pickle

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from PIL import Image
import matplotlib.pyplot as plt

class Real_Dataset(Dataset):
    def __init__(self, path, transform = False, train = True, ep = 1e-3):
        super(Real_Dataset).__init__()

        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()
        
        self.train = train
        self.alpha = 1000
            
        self.label = ['2S1', 'BMP2', 'BRDM2', 'BTR70', 'ZSU234']

        if train:
            self.dir_t = 'train'
        else:
            self.dir_t = 'test'

        # Data Path
        self.data_path = []
        self.data_label = []
        self.file_name = []
        for label in self.label:
            path2data = join(path, self.dir_t, label)
            for file_name in os.listdir(path2data):
                data_path = join(path2data, file_name)

                self.data_path.append(data_path)
                self.data_label.append(label)
                self.file_name.append(file_name)

    # Cropped Img
    def __getitem__(self, index):
        
        label = self.data_label[index]

        file_name = self.file_name[index]

        with open(self.data_path[index], 'rb') as f:
            data = pickle.load(f)

        data = abs(data['image'])
        data = data / np.max(data)
        data = np.log10(data * self.alpha + 1) / np.log10(self.alpha + 1)
        # h, w = data.shape

        # if h >= 64 or w>= 64:
        #     data = data[int(np.round(h/2 - 32)):int(np.round(h/2 + 32)), int(np.round(w/2 - 32)):int(np.round(w/2 + 32))]
        #     h = 64
        #     w = 64

        # zero = -1*torch.ones([64, 64])
        img = self.transform(data)
        # zero[int(np.round(32-h/2)):int(np.round(32+h/2)), int(np.round(32-w/2)):int(np.round(32+w/2))] = img
        # zero = zero.view(1, 64, 64)

        # return zero.type(torch.float32), label, file_name
        return img.type(torch.float32), label, file_name
    
    def __len__(self):
        return len(self.data_path)
    

class Synth_Dataset(Dataset):
    def __init__(self, path, transform = False, train = True, ep = 1e-3):
        super(Synth_Dataset).__init__()

        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()
        
        self.train = train
        self.alpha = 1000

        self.label = ['2S1', 'BMP2', 'BRDM2', 'BTR70', 'ZSU234']

        if train:
            self.dir_t = 'train'
        else:
            self.dir_t = 'test'


        # Data Path
        self.data_path = []
        self.data_label = []
        self.file_name = []
        for label in self.label:
            path2data = join(path, self.dir_t, label)
            for file_name in os.listdir(path2data):
                data_path = join(path2data, file_name)

                self.data_path.append(data_path)
                self.data_label.append(label)
                self.file_name.append(file_name)
    
    # Cropped Img
    def __getitem__(self, index):
        
        label = self.data_label[index]

        file_name = self.file_name[index]

        with open(self.data_path[index], 'rb') as f:
            data = pickle.load(f)

        data = abs(data['image'])
        data = data / np.max(data)
        data = np.log10(data * self.alpha + 1) / np.log10(self.alpha + 1)
        # h, w = data.shape

        # if h >= 64 or w>= 64:
        #     data = data[int(np.round(h/2 - 32)):int(np.round(h/2 + 32)), int(np.round(w/2 - 32)):int(np.round(w/2 + 32))]
        #     h = 64
        #     w = 64

        # zero = -1*torch.ones([64, 64])
        img = self.transform(data)
        # zero[int(np.round(32-h/2)):int(np.round(32+h/2)), int(np.round(32-w/2)):int(np.round(32+w/2))] = img
        # zero = zero.view(1, 64, 64)

        # return zero.type(torch.float32), label, file_name
        return img.type(torch.float32), label, file_name
    
    def __len__(self):
        return len(self.data_path)
    
    
    def __len__(self):
        return len(self.data_path)
    
if __name__ == '__main__':
    
    path = './Data/SAR Data/SAMPLE/results/CycleGAN_230817_ep1e-6_clip_10/refine/2s1/2s1_refine_A_elevDeg_015_azCenter_016_22_serial_b01.png'
    img = Image.open(path).convert('L')
    # img = plt.imread(path)
    plt.imshow(img, cmap = 'gray')
    plt.savefig('k.png')
    print(np.array(img).shape)
    print('a')
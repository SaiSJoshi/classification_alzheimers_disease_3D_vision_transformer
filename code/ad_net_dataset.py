import torch
from torch.utils import data as torch_data

import SimpleITK as sitk

import numpy as np
import glob, re, os

def load_files(path):
    all_imgs = []
    for root, subdirectories, files in os.walk(path):
        for file_path in files:
            if file_path == '.DS_Store' or file_path.endswith('.gz'):
                continue
        
            img_path = os.path.join(root, file_path)
            # Read the image in and print its size and display one slice
            img = sitk.ReadImage(img_path)
            np_img = sitk.GetArrayFromImage(img)
            print(np_img.shape, img_path)
            all_imgs.append(np_img)
    return all_imgs


def split_train_test(data, train_ratio=0.8):
    """splits the data passed to this method 
       into a list of elements specified ratio
    """
    train_split_pos = int(train_ratio*len(data))+1
    train_data = data[0:train_split_pos]
    test_data = data[train_split_pos:]
    return train_data, test_data


class ADNetDataset(torch_data.Dataset):
    def __init__(self, AD_dir_path='data/ad_train', CN_dir_path='data/cn_train'):
        super().__init__()
        ad_imgs = load_files(AD_dir_path)        
        cn_imgs = load_files(CN_dir_path)
        
        self.X = torch.tensor(np.array(ad_imgs + cn_imgs))
        # self.X = self.X.reshape(-1, 1, 182, 182, 218)
        self.X = self.X.reshape(-1, 1, 128, 128, 64)
        
        self.X = self.X[:,:,:,:,:210]
        y_ad = torch.zeros(len(ad_imgs))
        y_cn = torch.ones(len(cn_imgs))
        self.Y = torch.concat((y_ad, y_cn))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.Y[idx]}
## source: https://towardsdatascience.com/medical-images-segmentation-using-keras-7dc3be5a8524
## https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55

import os
import torch
import nibabel as nib
from torch.utils import data
import numpy as np

## create dataset to be used by the data loader
## pay attention: input and target list must have same order (input/target have the same name) in order to achieve the correct mapping

class DataSet(data.Dataset):
    def __init__(self,
                 img_path, label_path,
                 transform=None
                 ):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_id = self.img_path[index]
        label_id = self.label_path[index]
        img = nib.load(img_id)
        img = img.get_fdata()
        img = img.astype(float)
        label = nib.load(label_id)
        label = label.get_data()
        label = label.astype(float)

        ##TODO patch ausschneiden (128 x 128), patch size als parameter übergeben, sodass man verschiedene grössen testen kann
        ##maske ausschneiden an der gleichen Stelle
        ##grösse muss sicher so gross sein, dass nicht nur knochen vorhanden ist

        if self.transform is not None:
            img, label = self.transform(img, label)

        #Type casting
        img, label = torch.from_numpy(img).type(self.inputs_dtype), torch.from_numpy(label).type(self.targets_dtype)

        print(f'img = shape: {img.shape}; type: {img.dtype}')
        print(f'img = min: {img.min()}; max: {img.max()}')
        print(f'label = shape: {label.shape}; class: {label.unique()}; type: {label.dtype}')

        print(img.shape)
        print(label.shape)

        return img, label







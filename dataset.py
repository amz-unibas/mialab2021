import os
import nibabel as nib
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu

class DataSet(data.Dataset):
    def __init__(self,
                 img_path, label_path,
                 img_width, img_height,
                 transform=None,
                 ):
        self.img_path = img_path
        self.label_path = label_path
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform
        self.images = os.listdir(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.images[index])
        label_path = os.path.join(self.label_path, self.images[index]).replace(".nii.gz", "-label.nii.gz")

        img = nib.load(img_path)
        img = img.get_data()

        label = nib.load(label_path)
        label = label.get_data()
        label = label.astype('float32')
        label = label[:,:,0]

        if img.ndim > 3:
            ##check which image is the correct one since there are 2
            img = img[:, :, :, 0]
            # print("Adjusted dimensions: ", img.shape)

        ##albu Normalization
        normalize_transform = albu.Compose(
            [
                albu.Normalize(mean=0, std=1, max_pixel_value=np.amax(img))
            ]
        )
        img_d = normalize_transform(image=img, mask=label)
        img_norm = img_d["image"]
        label_norm = img_d["mask"]


        ##TOIMPROVE: maybe padd with mean value of the image
        ##add padding
        pad_img = np.zeros((self.img_width, self.img_height, 1))
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img_norm
        pad_label = np.zeros((self.img_width, self.img_height))
        pad_label[:label.shape[0], :label.shape[1]] = label_norm

        # adjust to NCHW format --> use ToTensorV2 (in main.py)
        # pad_img = pad_img.transpose(2, 1, 0)
        # pad_label = pad_label.transpose(2, 1, 0)

        if self.transform is not None:
            augmentations = self.transform(image=pad_img, mask=pad_label)
            pad_img = augmentations["image"]
            pad_label = augmentations["mask"]

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(label)
        # plt.show()
        # print("Shape img", pad_img.shape)
        # print("Shape label", pad_label.shape)

        return pad_img, pad_label
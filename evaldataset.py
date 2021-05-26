import os
import nibabel as nib
from torch.utils import data
import numpy as np
import albumentations as albu


class EvalDataSet(data.Dataset):
    heights = []
    widths = []
    def __init__(self,
                 img_path,
                 img_width, img_height,
                 transform=None,
                 ):
        self.img_path = img_path
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform
        self.images = os.listdir(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.images[index])
        img = nib.load(img_path)
        img = img.get_data()

        if img.ndim > 3:
            ##check which image is the correct one since there are 2
            img = img[:, :, :, 0]


        ##save h,w information for later
        self.heights.append(img.shape[0])
        self.widths.append(img.shape[1])

        ##albu Normalization
        normalize_transform = albu.Compose(
            [
                albu.Normalize(mean=0, std=1, max_pixel_value=np.amax(img))
            ]
        )
        img_d = normalize_transform(image=img)
        img_norm = img_d["image"]

        ##add padding
        pad_img = np.zeros((self.img_width, self.img_height, 1))
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img_norm

        if self.transform is not None:
            augmentations = self.transform(image=pad_img)
            pad_img = augmentations["image"]

        # plt.imshow(img)
        # plt.show()
        # print("Shape img", pad_img.shape)
        # pad_img.numpy()
        # pad_org = np.resize(pad_img, (3400, 3000, 1))
        # imaage = pad_org[:self.widths[0], :self.heights[0], :]
        # print("original shape: ", imaage.shape)

        return pad_img

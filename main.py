## source: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
from dataset_v1 import DataSet

import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib

def main():

    # define paths
    root = pathlib.Path.cwd() / 'data'

    # input and target files
    # imgs_path = get_filenames_of_path(root / 'images')
    # label_path = get_filenames_of_path(root / 'labels')

    ##TODO: get same random items for labels and images for training/testing
    # random seed
    random_seed = 42

    # split dataset into training set and validation set
    train_size = 0.8  # 80:20 split

    # train_images, test_images = train_test_split(
    #     imgs_path,
    #     random_state=random_seed,
    #     train_size=train_size,
    #     shuffle=True)
    #
    # train_labels, test_labels = train_test_split(
    #     label_path,
    #     random_state=random_seed,
    #     train_size=train_size,
    #     shuffle=True)

    train_imgs = get_filenames_of_path(root / 'train/images')
    test_imgs = get_filenames_of_path(root / 'test/images')
    train_labels = get_filenames_of_path(root / 'train/labels')
    test_labels = get_filenames_of_path(root / 'test/labels')

    train_imgs.sort()
    train_labels.sort()
    test_imgs.sort()
    test_labels.sort()

    img_height = 3000
    img_width = 3000

    training_dataset = DataSet(img_path=train_imgs, label_path=train_labels,
                               img_height=img_height, img_width=img_width,
                               transform=None)
    testing_dataset = DataSet(img_path=test_imgs, label_path=test_labels,
                              img_height=img_height, img_width=img_width,
                              transform=None)

    training_dataloader = DataLoader(dataset=training_dataset,
                                 batch_size=2,
                                 shuffle=True)

    testing_dataloader = DataLoader(dataset=testing_dataset,
                                 batch_size=2,
                                 shuffle=True)


    img, label = next(iter(training_dataloader))

    print(f'img = shape: {img.shape}; type: {img.dtype}')
    print(f'img = min: {img.min()}; max: {img.max()}')
    print(f'label = shape: {label.shape}; class: {label.unique()}; type: {label.dtype}')

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

if __name__ == "__main__":
    main()

# Source https://www.youtube.com/watch?v=IHq1t7NxS8k

import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import time


from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters

GPU_ID = 2
LEARNING_RATE = 1e-4
DEVICE = "cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 340
PAD_HEIGHT = 3000
PAD_WIDTH = 3400
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_LABEL_DIR = "data/train/labels/"
TEST_IMG_DIR = "data/test/images/"
TEST_LABEL_DIR = "data/test/labels/"
current_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

result_path = os.path.join("/results", current_date)
writer = SummaryWriter()


def train_fn(loader, model, optimizer, loss_fn, scaler, idx):
    #progress bar
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # tensorboard
        writer.add_scalar('cross entropy loss ', loss.item(), idx)

        if idx % 25 == 0:
            writer.add_images("input images", targets.detach().cpu(), idx)
            writer.add_images("estimated labels", predictions, idx)



def main():
    train_transform = albu.Compose(
        [
            albu.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            albu.Rotate(limit=10, p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.1),
            albu.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
    )

    test_transforms = albu.Compose(
        [
            albu.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            albu.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    #binary cross entropy
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    idx = 0

    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_LABEL_DIR,
        TEST_IMG_DIR,
        TEST_LABEL_DIR,
        PAD_WIDTH,
        PAD_HEIGHT,
        BATCH_SIZE,
        train_transform,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()


    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, idx)
        idx += 1
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(test_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            test_loader, model, folder="saved_images/", device=DEVICE
        )
        writer.flush()
        # use sleep to show the training
        time.sleep(0.2)

    writer.close()


if __name__ == "__main__":
    main()

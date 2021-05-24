# Source https://www.youtube.com/watch?v=IHq1t7NxS8k

import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from torch.utils.tensorboard import SummaryWriter
import time
from omegaconf import OmegaConf


from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    evaluate, get_eval_loader)


GPU_ID = 2
DEVICE = "cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_LABEL_DIR = "data/train/labels/"
TEST_IMG_DIR = "data/test/images/"
TEST_LABEL_DIR = "data/test/labels/"
EVAL_IMG_DIR = "data/eval/images/"
TEST_BONUS = False

writer = SummaryWriter()


def train_fn(loader, model, optimizer, loss_fn, scaler, idx):
    # progress bar
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data.float())
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # tensorboard
        writer.add_scalar('loss ', loss.item(), idx)

def main():

    ##config
    # get the cli commands
    cli_conf = OmegaConf.from_cli()
    # load the config file
    if cli_conf.config_filename is not None:
        cfg_conf = OmegaConf.load(cli_conf.config_filename)
        # merge the commands
        cfg = OmegaConf.merge(cfg_conf, cli_conf)
    else:
        cfg = cli_conf

    if len(cfg) == 0 and len(cli_conf) == 0:
        cfg = OmegaConf.load("config.yaml")

    if TEST_BONUS:
        print("testing circle detection and co.")
        ##TODO: load images, from tensor to np.array

        ##TODO: detect circle on original image, save diameter

        ##TODO: detect 4 objects on prediction image

        #   calculate least distance between the 2 object pairs

        #   calculate the max width of each object

    ##transforms
    train_transform = albu.Compose(
        [
            albu.Resize(height=cfg.images.img_h, width=cfg.images.img_w),
            albu.Rotate(limit=10, p=0.5),
            #albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.Blur(blur_limit=5, always_apply=False, p=0.5),
            #TODO label is float64, should be float32 to work for the brightness/contrast
            #albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5),
            ToTensorV2(),
        ],
    )

    test_transforms = albu.Compose(
        [
            albu.Resize(height=cfg.images.img_h, width=cfg.images.img_w),
            ToTensorV2(),
        ],
    )

    eval_transforms = albu.Compose(
        [
            #try to keep the original size of the inout images, otherwise uncomment
            albu.Resize(height=cfg.images.img_h, width=cfg.images.img_w),
            ToTensorV2(),
        ],
    )

    ##model parameters
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    idx = 0
    scaler = torch.cuda.amp.GradScaler()
    if cfg.model.load_model:
        load_checkpoint(torch.load(cfg.model.load_name, map_location=torch.device('cpu')), model)


    if cfg.model.eval_mode:
        eval_loader = get_eval_loader(
            EVAL_IMG_DIR,
            cfg.images.pad_w,
            cfg.images.pad_h,
            cfg.training.batch_size,
            eval_transforms,
            cfg.training.num_workers,
            PIN_MEMORY,
        )
        evaluate(eval_loader, model, writer, device=DEVICE)
        writer.flush()
        # use sleep to show the training
        time.sleep(0.2)


    else:
        ##loaders
        train_loader, test_loader = get_loaders(
            TRAIN_IMG_DIR,
            TRAIN_LABEL_DIR,
            TEST_IMG_DIR,
            TEST_LABEL_DIR,
            cfg.images.pad_w,
            cfg.images.pad_h,
            cfg.training.batch_size,
            train_transform,
            test_transforms,
            cfg.training.num_workers,
            PIN_MEMORY,
        )
        check_accuracy(test_loader, model, writer, device=DEVICE)

        for epoch in range(cfg.training.num_epochs):
            train_fn(train_loader, model, optimizer, loss_fn, scaler, idx)
            idx += 1
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, cfg.model.save_name)

            if idx % 5 == 0:
                # check accuracy
                check_accuracy(test_loader, model, writer, device=DEVICE)
                # print some examples to a folder
                save_predictions_as_imgs(test_loader, model, index=idx, folder="saved_images/", device=DEVICE)

            writer.flush()
            # use sleep to show the training
            time.sleep(0.2)

    writer.close()


if __name__ == "__main__":
    main()

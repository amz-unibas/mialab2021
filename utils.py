import torch
import torchvision
from dataset import DataSet
from torch.utils.data import DataLoader
import numpy as np


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_labeldir,
        test_dir,
        test_labeldir,
        img_width,
        img_height,
        batch_size,
        train_transform,
        test_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = DataSet(
        img_path=train_dir,
        label_path=train_labeldir,
        img_width=img_width,
        img_height=img_height,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_ds = DataSet(
        img_path=test_dir,
        label_path=test_labeldir,
        img_width=img_width,
        img_height=img_height,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader


def check_accuracy(loader, model, writer, loss_fn, device="cuda:2"):
    dice_score = 0
    model.eval()
    #no gradients
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x.float()))
            # convert all values > 0.5 to 1
            preds = (preds > 0.5).float()
            #TODO dice score for each image, not in total
            dice_score += calculate_dice_score(y, preds)

    print(f"Dice score: {dice_score / len(loader)}")

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        targets = y.float().unsqueeze(1).to(device)
        preds = torch.sigmoid(model(x.float()))
        # tensorboard
        ##TODO image back normalization (+min, /max)
        writer.add_images("input images", x.detach().cpu(), idx)
        writer.add_images("target labels", targets.detach().cpu(), idx)
        writer.add_images("estimated labels", preds.detach().cpu(), idx)

    model.train()

##TODO check implementation, seems like torch.sum(y) == 0 and torch.sum(preds) == 0 is often the case?
def calculate_dice_score(y, preds):
    intersection = torch.sum(preds * y)
    if torch.sum(y) == 0 and torch.sum(preds) == 0:
        return 1
    return 2 * intersection / (torch.sum(preds) + torch.sum(y))


def save_predictions_as_imgs(loader, model, index, folder="saved_images/", device="cuda:2"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}pred_{index}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{index}.png")

    model.train()

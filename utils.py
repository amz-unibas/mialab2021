import torch
import torchvision
from dataset import DataSet
from evaldataset import EvalDataSet
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2 as cv

def save_checkpoint(state, filename):
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
        num_workers,
        pin_memory,
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


def get_eval_loader(
        eval_dir,
        img_width,
        img_height,
        batch_size,
        eval_transform,
        num_workers,
        pin_memory,
):
    eval_ds = EvalDataSet(
        img_path=eval_dir,
        img_width=img_width,
        img_height=img_height,
        transform=eval_transform,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return eval_loader

def check_accuracy(loader, model, writer, device):
    dice_score = 0
    model.eval()
    # no gradients
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        targets = y.float().unsqueeze(1).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x.float()))
            preds = (preds > 0.5).float()

        dice_score += calculate_dice_score(y, preds)
        # tensorboard
        writer.add_images("input images", x.detach().cpu(), idx)
        writer.add_images("target labels", targets.detach().cpu(), idx)
        writer.add_images("estimated labels", preds.detach().cpu(), idx)

    print(f"Dice score: {dice_score /   len(loader)}")
    model.train()


##calculate DICE similarity
def calculate_dice_score(y, preds):
    intersection = torch.sum(preds * y)
    return 2 * intersection / (torch.sum(preds) + torch.sum(y))


def save_predictions_as_imgs(loader, model, index, folder, device):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x.float()))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}pred_{index}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{index}.png")

    model.train()


def evaluate(loader, model, device, cfg):
    model.eval()
    # no gradients
    for idx, (x, y, z, w, v) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x.float()))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, "pred_all_images.png")
        # tensor to numpy array
        pred_np = preds.detach().cpu().numpy()

        #save as png
        for img in range(len(loader.dataset.images)/cfg.training.num_epochs):

            print("shape: ", pred_np.shape)
            pred = pred_np[img]

            # adjust to WHC, maybe HWC (1,2,0)
            trans_preds = pred.transpose(1, 2, 0)

            ##resize, remove padding
            #TODO try interpolation methods
            #preds_resized = cv.resize(trans_preds, (cfg.images.pad_h, cfg.images.pad_w), interpolation=cv.INTER_LINEAR)
            #preds_resized = cv.resize(trans_preds, (cfg.images.pad_h, cfg.images.pad_w), interpolation=cv.INTER_CUBIC)
            preds_resized = cv.resize(trans_preds, (cfg.images.pad_h, cfg.images.pad_w), interpolation=cv.INTER_NEAREST)
            predicitions = preds_resized[:z[img], :y[img]]


    #     #TODO BONUS
    #     ##TODO: load images, from tensor to np.array
    #
    #     ##TODO: detect circle on original image, save diameter
    #
    #     ##TODO: detect 4 objects on prediction image

            # save as nifti
            affine = w[img]
            ni_preds = nib.nifti1.Nifti1Image(predicitions, affine)
            nib.save(ni_preds, "predictions/label-" + v[img])



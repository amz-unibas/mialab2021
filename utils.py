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
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x.float()))
            # convert all values > 0.5 to 1
            preds = (preds > 0.5).float()
            dice_score += calculate_dice_score(y, preds)

    print(f"Dice score: {dice_score / len(loader)}")

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        targets = y.float().unsqueeze(1).to(device)
        preds = torch.sigmoid(model(x.float()))
        # tensorboard
        writer.add_images("input images", x.detach().cpu(), idx)
        writer.add_images("target labels", targets.detach().cpu(), idx)
        writer.add_images("estimated labels", preds.detach().cpu(), idx)

    model.train()


def evaluate(loader, model, writer, device, cfg):
    model.eval()
    # no gradients
    for idx, (x, y, z, w) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x.float()))
            preds = (preds > 0.5).float()
        # tensorboard
        # writer.add_images("input images", x.detach().cpu(), idx)
        # writer.add_images("estimated labels", preds.detach().cpu(), idx)
        print("index: ", idx)
        #save as png
        torchvision.utils.save_image(preds, f"pred_{loader.dataset.images[idx]}.png")

        # ##tensor to numpy array
        pred_np = preds.detach().cpu().numpy()
        preds_np = pred_np[0,:,:,:]

        # adjust to WHC, maybe HWC (1,2,0)
        trans_preds = preds_np.transpose(2, 1, 0)

        ##resize, remove padding
        preds_resized = cv.resize(trans_preds, (cfg.images.pad_w, cfg.images.pad_h), interpolation=cv.INTER_LINEAR)

        predicitions = preds_resized[:y[idx], :z[idx]]

        ##save as nifti
        affine = w[idx]
        #header = v[idx]
        xform = np.eye(4)
        ni_preds = nib.nifti1.Nifti1Image(predicitions, affine)
        ni_preds_1= nib.nifti1.Nifti1Image(predicitions, None)
        nif_preds_2 = nib.nifti1.Nifti1Image(predicitions, xform)
        nib.save(ni_preds, "predictions/label-" + loader.dataset.images[idx])
        nib.save(ni_preds_1, "predictions/label1-" + loader.dataset.images[idx])
        nib.save(nif_preds_2, "predictions/label2-" + loader.dataset.images[idx])

    model.train()


##calculate DICE similarity
def calculate_dice_score(y, preds):
    intersection = torch.sum(preds * y)
    if torch.sum(y) == 0 and torch.sum(preds) == 0:
        return 1
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

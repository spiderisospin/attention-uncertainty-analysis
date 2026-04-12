# -*- coding: utf-8 -*-

from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
from src.grad_cam import *

#Plot learning curves for ResNet and LeVit

def learning_curve(resnet_history, levit_history, baseline_score):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ResNet loss
    axes[0, 0].plot(resnet_history["train_loss"], marker="o", label="train")
    axes[0, 0].plot(resnet_history["val_loss"], marker="o", label="validation")
    axes[0, 0].set_title("ResNet18 Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE loss")
    axes[0, 0].legend()

    # ResNet RMSE
    axes[0, 1].plot(resnet_history["val_rmse"], marker="o", label="validation RMSE")
    axes[0, 1].axhline(baseline_score, linestyle="--", label="baseline RMSE")
    axes[0, 1].set_title("ResNet18 RMSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].legend()

    # LeViT loss
    axes[1, 0].plot(levit_history["train_loss"], marker="o", label="train")
    axes[1, 0].plot(levit_history["val_loss"], marker="o", label="validation")
    axes[1, 0].set_title("LeViT-128s Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE loss")
    axes[1, 0].legend()

    # LeViT RMSE
    axes[1, 1].plot(levit_history["val_rmse"], marker="o", label="validation RMSE")
    axes[1, 1].axhline(baseline_score, linestyle="--", label="baseline RMSE")
    axes[1, 1].set_title("LeViT-128s RMSE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

#Grad-CAM plotting

def plot_gradcam(model, cam, df, img_dir, transform, rmse_np, target_cols):
    """
    This function plots two high-RMSE and two low-RMSE samples with their corresponding
    Grad-Cam heatmaps. The original images are plotted left and Grad-CAM heatmaps on the right.

    Parameters
        model - torch.nn.Module
            Neural network used for model training.
        cam - GradCAM
            GradCAM object used to create heatmap.
        df - pd.DataFrame
            DataFrame with file names and target columns.
        img_dir - str
            Path to images directory.
        transform - Callable
            Transformations applied per image.
        rmse_np - array
            Array containing RMSE values for each sample.
        target_cols - list
            List containing the columns names of our targets.

    Returns
        None
    """
    #find worst and best RMSE image indexes
    worst_i = np.argsort(-rmse_np)[:2]
    best_i = np.argsort(rmse_np)[:2]
    chosen_i = list(worst_i) + list(best_i)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    #plotting Grad-CAM
    for k, i in enumerate(chosen_i):
        row_i = k // 2
        col_i = (k % 2) * 2

        rgb_img, grayscale_cam, target_class = compute_gradcam(model, cam, df, img_dir, transform, i)

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[row_i, col_i].imshow(rgb_img)
        axes[row_i, col_i].set_title(f"RMSE={rmse_np[i]:.3f}")

        axes[row_i, col_i + 1].imshow(visualization)
        axes[row_i, col_i + 1].set_title(f"Grad-CAM: {target_cols[target_class]}")

    plt.tight_layout()
    plt.show()
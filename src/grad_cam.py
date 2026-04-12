# -*- coding: utf-8 -*-

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
from src.data_processing import *
from src.class_models import *
from src.train_models import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

eps = 1e-8 #constant to avoid logarithm divergences

#Obtain probabilistic predictions

def get_predictions(model, df, img_dir, transform, target_cols, batch_size):
    """
    This function initializes batches and loops over them. Our model is applied to the
    data and converts to probabilities using the sigmoid. Furthermore, these predictions
    are concatenated into two tensors which are returned.

    Parameters
        model - torch.nn.Module
            Neural network used for model training.
        df - pd.DataFrame
            DataFrame with file names and target columns.
        img_dir - str
            Path to images directory.
        transform - Callable
            Transformations applied per image.
        target_cols - list
            List containing the columns names of our targets.
        batch_size - int
            Amount of samples per batch.

    Returns
        probabilities - torch.Tensor
            Model probabilities outputted by sigmoid.
        targets - torch.Tensor
            True target tensor.
    """
    probabilities = []
    targets = []

    batches = create_batches(df, img_dir, transform, batch_size, target_cols, True)

    model.eval()
    with torch.no_grad(): #no gradients
        for images, batch_targets in batches:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu()

            probabilities.append(probs)
            targets.append(batch_targets)

    probabilities = torch.cat(probabilities, dim=0)
    targets = torch.cat(targets, dim=0)

    return probabilities, targets

#Create GradCAM heatmap

def compute_gradcam(model, cam, df, img_dir, transform, i):
    """
    This function loads an image located in dataframe df at index i. A forward pass is ran
    and a Grad-CAM heatmap is computed for the top prediction of the model. We return the
    resized image (normalized), heat map and predicted class index

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
        i - int
            Index selecting image.

        Returns
            resized_img - array
                Resized image for visualization, scaled between [0,1].
            heatmap - array
                Grad-CAM heatmap showing regions contributing most to prediction.
            target_class - int
                Index of model's best predicted class.
        """
    path = os.path.join(img_dir, df.iloc[i]["filename"])
    img = Image.open(path).convert("RGB")

    input_tensor = transform(img).unsqueeze(0).to(device)
    resized_img = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0]
        target_class = int(torch.argmax(probs).item())

    heatmap = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]

    return resized_img, heatmap, target_class

#Determine entropy of CAM heatmap per image

def CAM_entropy(model, cam, df, img_dir, transform):
    """
    This function looks over the main dataset and computes the Grad-CAM heatmap for each image.
    For each image, the entropy of their corresponding heatmap is computed.

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

    Returns
        np.array(cam_entropy_vals)
            Array containing entropy values of each image in the dataset.
    """
    cam_entropy_vals = []

    for i in range(len(df)):
        grayscale_cam = compute_gradcam(model, cam, df, img_dir, transform, i)[1]

        cam_map = grayscale_cam / (grayscale_cam.sum() + eps) #normalize
        cam_entropy = -np.sum(cam_map * np.log(cam_map + eps))
        cam_entropy_vals.append(cam_entropy)

    return np.array(cam_entropy_vals)

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
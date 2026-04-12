# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.data_processing import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#Train for one epoch

def train_epoch(model, target_cols, df, img_dir, transform, optimizer, device, batch_size):
    """
    This function trains our model for one epoch over the data set. The function calls the
    create_batches function to initialize batches, and subsequently iterates over the
    images and targets in batches. A forward pass is performed the loss is determined
    using our loss criterion. Backpropagation is used toupdate the parameters of our
    model and lastly a sigmoid activation function is applied on those.

    Parameters
        model - torch.nn.Module
            Neural network used for model training.
        target_cols - list
            List containing the columns names of our targets.
        df - pd.DataFrame
            DataFrame with file names and target columns.
        img_dir - str
            Path to images directory.
        transform - Callable
            Transformations applied per image.
        optimizer (torch.optim.Optimizer)
            Optimizer used for updating model parameters.
        device - torch.device
            device chosen to run on (GPU or CPU).
        batch_size - int
            Amount of samples per batch.

    Returns
        loss_avg - float
            Average training loss over entire training epoch.

    """
    criterion = nn.MSELoss() #MSE loss
    model.train() #training mode
    running_loss = 0.0
    n_batches = 0

    batches = create_batches(df, img_dir, transform, batch_size, target_cols, True)

    for images, targets in batches:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad() #reset previous gradients
        outputs = model(images)
        outputs = torch.sigmoid(outputs) #logits to probabilities

        loss = criterion(outputs, targets)
        loss.backward() #backprop
        optimizer.step() #update weights

        running_loss += loss.item()
        n_batches += 1

    loss_avg = running_loss / n_batches
    return loss_avg

#Validate for one epoch
def validate_epoch(model, target_cols, df, img_dir, transform, device, batch_size):
    """
    Here the validation of 'model' is taking place. This is being done by creating batches,
    performing a forward pass without updating parameters. Subsequently the loss and RMSE
    are computed and returned.

    Parameters
        model - torch.nn.Module
            Neural network used for model training.
        target_cols - list
            List containing the columns names of our targets.
        df - pd.DataFrame
            DataFrame with file names and target columns.
        img_dir - str
            Path to images directory.
        transform - Callable
            Transformations applied per image.
        device - torch.device
            device chosen to run on (GPU or CPU).
        batch_size - int
            Amount of samples per batch.

    Returns
        loss_avg - float
            Average training loss over entire validation epoch.
        rmse - float
            RMSE value determined over all samples and targets.
    """
    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0
    n_batches = 0
    preds_all = []
    targets_all = []

    batches = create_batches(df, img_dir, transform, batch_size, target_cols, True)

    with torch.no_grad(): #compute no gradients
        for images, targets in batches:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs) #logits to probabilities
            loss = criterion(probs, targets)

            running_loss += loss.item()
            n_batches += 1

            #store predictions and true
            preds_all.append(probs.cpu())
            targets_all.append(targets.cpu())

    loss_avg = running_loss / n_batches
    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    rmse = torch.sqrt(torch.mean((preds_all - targets_all) ** 2)).item()

    return loss_avg, rmse

#Train model

def train_model(model, model_name, target_cols, train_df, val_df, img_dir, train_transform,
                val_transform, epochs, learning_rate, batch_size):
    """
    This function trains the our model. The AdamW opimizer and cosine annealing learning
    rate scheduler are used for this purpose. For each epoch, we run the train_epoch and
    validate_epoch functions in order to find the losses. Subsequently, the scheduler is
    used in order to update the learning rate. We then save the loss statistics to our
    history dictionary and save the model weights if our validation RMSE improves. After
    all epochs ar ran, we return the statistics of our execution.

    Parameters
        model - torch.nn.Module
            Neural network used for model training.
        model_name - str
            Name of the model used.
        target_cols - list
            List containing the columns names of our targets.
        train_df - pd.DataFrame
            Train dataset.
        val_df - pd.DataFrame
            Validation dataset.
        img_dir - str
            Path to images directory.
        train_transform - Callable
            Transformations applied to train set images.
        val_transform - Callable
            Transformations applied to validation set images.
        epochs - int
            Number of epochs to train model over.
        learning_rate - float
            Learning rate used by optimizer.
        batch_size - int
            Amount of samples per batch.

    Return
        history - dict
            Train and Validation metrics per epoch.
        best_rmse - float
            Best validation RMSE.
        best_path - str
            Path of best model saved.
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #AdamW optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) #ConsineAnnealing

    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_rmse = np.inf
    best_path = f"best_{model_name}.pth"

    for epoch in range(epochs):
        #train and validation metrics
        train_loss = train_epoch(model, target_cols, train_df, img_dir, train_transform, optimizer, device, batch_size)
        val_loss, val_rmse = validate_epoch(model, target_cols, val_df, img_dir, val_transform, device, batch_size)

        scheduler.step() #update weights

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)

        print(f"{model_name}: Epoch {epoch+1}/{epochs} -- train_loss={train_loss:.4f} -- val_loss={val_loss:.4f} -- val_rmse={val_rmse:.4f}")

        if val_rmse < best_rmse:
            #update RMSE if better
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_path)

    return history, best_rmse, best_path
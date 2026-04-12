# -*- coding: utf-8 -*-

import torch
import pandas as pd
import timm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#Baseline model

def baseline_rmse(df, mean_target, target_cols):
    """
    Determine the RMSE for our baseline model. The baseline is defined as always
    predicting the mean value for every sample of the training set. This results
    in a constant RMSE prediction for all data elements

    Parameters
        df - pd.DataFrame
            DataFrame with file names and target columns.
        mean_target - torch.Tensor
            Tensor which contains the mean values of each target in the training set.
        target_cols - list
            List containing the columns names of our targets.

    Returns
        float
            Baseline RMSE value determined between targets and mean prediction across all samples.
    """
    values = df[target_cols].values
    y_true = torch.tensor(values, dtype=torch.float32)
    y_pred = mean_target.unsqueeze(0).repeat(len(df), 1)

    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return rmse.item()

#ResNet and LeVit-128 models
def resnet18(n_targets):
    model = timm.create_model("resnet18", pretrained=True, num_classes = n_targets)
    return model.to(device)

def levit_128s(n_targets):
    model = timm.create_model("levit_128s", pretrained=True, num_classes = n_targets)
    return model.to(device)
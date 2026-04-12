# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import random
import torch
import os


def create_batches(df, img_dir, transform, batch_size, target_cols, include_targets):
    """
    This function shuffles our data indexes, loads and opens the images and
    subsequently applies data augmentation to these files. Finally, these
    images are returned with their targets.

    Parameters
        df - pd.DataFrame
            DataFrame with file names and target columns.
        img_dir - str
            Path to images directory.
        transform - Callable
            Transformations applied per image.
        batch_size - int
            Amount of samples per batch.
        target_cols - list
            List containing the columns names of our targets.
        target_cols - list
            List containing target names.
        include_targets - bool
            Used to control whether targets are included(in train/val) or
            ID (test).

    Returns
        Iterator[Tuple[torch.Tensor, torch.Tensor]]
            Yields batches of images with their targets.
    """
    indexes = list(range(len(df)))
    random.shuffle(indexes)

    for i in range(0, len(indexes), batch_size):
        batch_index = indexes[i:i + batch_size]

        images = []
        second_output = [] #output may vary

        for j in batch_index:
            row = df.iloc[j]

            path = os.path.join(img_dir, row["filename"])
            image = Image.open(path).convert("RGB")
            image = transform(image)

            images.append(image)

            if include_targets:
                values = row[target_cols].values.astype(np.float32)
                second_output.append(torch.from_numpy(values))
            else:
                second_output.append(row["GalaxyID"])

        if include_targets:
            yield torch.stack(images), torch.stack(second_output)
        else:
            yield torch.stack(images), second_output
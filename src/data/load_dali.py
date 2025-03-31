"""
load_dali.py

This module contains functions for creating DALI data loaders and the associated image pipeline.
It assigns CPU affinity for DALI processing, converts textual labels to integer encodings,
and defines a public API (get_dali_loaders) to generate loaders for training, validation, and test splits.

The label mappings used for multi-task classification are now centralized and imported from config.py.
"""

import os
import pandas as pd

# Explicitly assign all CPU cores to DALI to prevent potential binding errors
os.environ["DALI_NUM_THREADS"] = "16"
os.environ["DALI_AFFINITY_MASK"] = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

import psutil
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_torch

import numpy as np
import torch
from random import shuffle
import config

# -----------------------------------------------------------------------------
# CPU Affinity Setup
# -----------------------------------------------------------------------------


def set_cpu_affinity_all():
    """
    Ensures this process can run on all available CPU cores.
    Some HPC clusters or container environments may restrict CPU visibility.
    This function sets the affinity to all available cores.
    """
    p = psutil.Process(os.getpid())
    total_cpus = psutil.cpu_count()
    p.cpu_affinity(list(range(total_cpus)))


# -----------------------------------------------------------------------------
# Utility Functions to Convert String Labels to Int Encodings
# -----------------------------------------------------------------------------


def map_label_to_int(label, label_mapping_dict):
    """
    Converts a single textual label (e.g. "mild") into an integer ID (e.g. 1)
    using the provided dictionary mapping.

    Parameters
    ----------
    label : str
        The label to be converted (e.g. 'mild').
    label_mapping_dict : dict
        A mapping from integer to string (e.g. {0: 'little_or_none', 1: 'mild', ...}).

    Returns
    -------
    int or None
        The corresponding integer ID, or None if no match is found.
    """
    for key, value in label_mapping_dict.items():
        if label == value:
            return key
    return None


def map_labels(row, label_mappings):
    """
    Combines four task labels into a single integer by placing each task's integer ID
    in a different 'decimal place' of a base-10 number.

    For example, if:
      - damage_severity -> mild (ID = 1)
      - humanitarian -> not_humanitarian (ID = 2)
      - informative -> informative (ID = 1)
      - disaster_types -> flood (ID = 2)
    we form: 1*1000 + 2*100 + 1*10 + 2 = 1212

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame with columns 'damage_severity', 'humanitarian',
        'informative', and 'disaster_types'.
    label_mappings : dict of dict
        Nested dictionary of label mappings. Now, use the centralized LABEL_MAPPINGS.

    Returns
    -------
    int
        A single integer representing the combined label.
    """
    damage = (
        map_label_to_int(row["damage_severity"], label_mappings["damage_severity"])
        * 1000
    )
    humanitarian = (
        map_label_to_int(row["humanitarian"], label_mappings["humanitarian"]) * 100
    )
    informative = (
        map_label_to_int(row["informative"], label_mappings["informative"]) * 10
    )
    disaster = map_label_to_int(row["disaster_types"], label_mappings["disaster_types"])
    return damage + humanitarian + informative + disaster


# -----------------------------------------------------------------------------
# Primary Public API: get_dali_loaders
# -----------------------------------------------------------------------------


def get_dali_loaders(data_dir, batch_size, num_threads, device_id, phase):
    """
    Creates three DALI data loaders for the 'train', 'val', and 'test' splits,
    returning them as a tuple. Each loader yields batches containing:
      - 'data': Batch of processed images.
      - 'labels': Batch of combined integer labels.

    Additionally, the returned tuple includes a list of file paths corresponding to
    the samples in the pipeline (for val/test splits).

    Parameters
    ----------
    data_dir : str
        Base directory containing the dataset TSV files and image subfolders.
    batch_size : int
        Number of samples per batch.
    num_threads : int
        Number of CPU threads to allocate for DALI processing.
    device_id : int
        GPU device ID for the pipeline.
    phase : str
        Typically 'train', 'val', or 'test'. (Used to control augmentations and shuffling.)

    Returns
    -------
    tuple
      (
        (train_loader, train_batches, train_files),
        (val_loader,   val_batches,   val_files),
        (test_loader,  test_batches,  test_files),
      )
    """
    set_cpu_affinity_all()
    num_threads = max(1, num_threads)

    # Use config's versioned paths instead of constructing them
    data_paths = config.get_data_paths()
    train_tsv = data_paths["train"]
    val_tsv = data_paths["val"]
    test_tsv = data_paths["test"]

    required_columns = [
        "image_path",
        "damage_severity",
        "informative",
        "humanitarian",
        "disaster_types",
    ]

    # Read TSV files with only required columns
    train_df = pd.read_table(train_tsv, usecols=required_columns)
    val_df = pd.read_table(val_tsv, usecols=required_columns)
    test_df = pd.read_table(test_tsv, usecols=required_columns)

    # For each split, build the full image path and compute the combined label.
    # Here we use the centralized LABEL_MAPPINGS.
    for df in [train_df, val_df, test_df]:
        df["full_img_path"] = df["image_path"].apply(
            lambda x: os.path.join(data_dir, x)
        )
        df["combined_label"] = df.apply(
            lambda row: map_labels(row, config.LABEL_MAPPINGS), axis=1
        )

    # Build individual loaders.
    train_loader, train_batches, train_files = get_dali_loader(
        train_df, batch_size, num_threads, device_id, phase="train"
    )
    val_loader, val_batches, val_files = get_dali_loader(
        val_df, batch_size, num_threads, device_id, phase="val"
    )
    test_loader, test_batches, test_files = get_dali_loader(
        test_df, batch_size, num_threads, device_id, phase="test"
    )

    return (
        (train_loader, train_batches, train_files),
        (val_loader, val_batches, val_files),
        (test_loader, test_batches, test_files),
    )


def get_dali_loader(df, batch_size, num_threads, device_id, phase):
    """
    Builds and initializes the DALI pipeline for a given DataFrame of image paths and integer labels.
    Returns a tuple (dali_loader, total_batches, files).

    The 'files' list contains the image paths in the same order as processed by the pipeline.
    For training, the file paths are omitted (set to None) for speed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'full_img_path' and 'combined_label'.
    batch_size : int
        Number of samples per batch.
    num_threads : int
        Number of CPU threads for DALI processing.
    device_id : int
        GPU device ID for the pipeline.
    phase : str
        'train', 'val', or 'test'. For train, shuffling is enabled; for val/test, not.

    Returns
    -------
    tuple
      (dali_loader, total_batches, files)
    """
    set_cpu_affinity_all()

    files = df["full_img_path"].tolist()
    labels = df["combined_label"].tolist()

    # Shuffle only if phase is 'train'
    shuffle_after = phase == "train"

    pipe = image_pipeline(
        files, labels, batch_size, num_threads, device_id, phase, shuffle_after
    )
    pipe.build()

    total_batches = (len(files) + batch_size - 1) // batch_size

    dali_loader = dali_torch.DALIGenericIterator(
        pipe,
        ["data", "labels"],
        last_batch_policy=dali_torch.LastBatchPolicy.PARTIAL,
        auto_reset=True,
        reader_name="Reader",
    )

    ret_files = None if phase == "train" else files

    return dali_loader, total_batches, ret_files


@dali.pipeline_def(
    batch_size=32,
    num_threads=5,
    device_id=0,
    prefetch_queue_depth=4,
    set_affinity=False,
)
def image_pipeline(
    files, labels, batch_size, num_threads, device_id, phase, shuffle_after
):
    """
    Standard 2-output pipeline: (images, combined_labels).

    Reads 'files' and 'labels' with the specified shuffle behavior. Applies random crops
    for training and center crops for validation/test.

    Parameters
    ----------
    files : list[str]
        List of image file paths.
    labels : list[int]
        List of combined integer labels.
    batch_size, num_threads, device_id : int
        Standard pipeline arguments.
    phase : str
        'train', 'val', or 'test' (determines augmentation behavior).
    shuffle_after : bool
        True for training; False for validation/test.

    Returns
    -------
    tuple
      (images, combined_labels)
    """
    set_cpu_affinity_all()

    jpegs, combined_labels = fn.readers.file(
        files=files,
        labels=labels,
        prefetch_queue_depth=2,
        shuffle_after_epoch=shuffle_after,
        name="Reader",
    )

    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.DALIImageType.RGB,
        hw_decoder_load=0.9,
    )
    images = fn.resize(
        images,
        resize_x=256,
        resize_y=256,
        interp_type=types.DALIInterpType.INTERP_LINEAR,
        antialias=False,
    )

    if phase == "train":
        images = fn.crop_mirror_normalize(
            images,
            crop_h=224,
            crop_w=224,
            crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
            crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=types.FLOAT,
            mirror=fn.random.coin_flip(probability=0.5),
        )
    else:
        images = fn.crop_mirror_normalize(
            images,
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=types.FLOAT,
        )

    return images, combined_labels

import torch
import os
import torch.nn as nn
import math
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy.typing as npt
import numpy as np
from typing import IO, Any, BinaryIO

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    max_start_idx = len(dataset) - context_length
    # 随机选择batch_size个起始位置
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    labels = np.zeros((batch_size, context_length), dtype=np.int64)
    for i, start_idx in enumerate(start_indices):
        inputs[i] = dataset[start_idx: start_idx+context_length]
        labels[i] = dataset[start_idx+1: start_idx+context_length+1]
    
    inputs_tensor = torch.from_numpy(inputs).long().to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    return inputs_tensor, labels_tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']



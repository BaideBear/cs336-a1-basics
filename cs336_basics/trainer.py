#参考实现：https://github.com/mocibb/cs336/blob/main/assignment1-basics/train_loop.py

# coding=utf-8
# Copyright (c) 2025 mocibb (mocibb@163.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
from dataclasses import dataclass
from .loss import cross_entropy
from .utils import lr_cosine_schedule, gradient_clipping
from .optimizer import AdamW
import wandb 
from datetime import datetime
from .model import Transformer
from .tokenizer import Tokenizer
from tqdm import tqdm
import time


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

@dataclass
class PretrainedConfig():
    project_name: str
    vocab_path: str
    merges_path: str
    special_tokens: list[str]
    train_path: str
    valid_path: str

    batch_size: int = 32
    vocab_size: int = 50257 #adapted to the true size
    context_len: int = 256
    d_model: int = 512
    d_ff: int = 1344
    rope_theta: float = 100000
    num_layers: int = 4
    num_heads: int = 16
    use_compile: bool = False

    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    warmup_steps: int = 500   # 10% of total_steps
    total_steps: int = 5000

    log_freq: int = 20
    eval_freq: int = 50
    eval_batch: int = 10
    checkpoint_freq: int = 50
    checkpoint_dir: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train(step, dataset: npt.NDArray, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config):
    inputs, targets = get_batch(dataset, config.batch_size, config.context_len, config.device)
    model.train()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), config.gradient_clipping)
    optimizer.step()
    return loss.item()

def evaluate(dataset: npt.NDArray, model: torch.nn.Module, config):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(config.eval_batch):
            inputs, targets = get_batch(dataset, config.batch_size, config.context_len, config.device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            losses.append(loss)
    return sum(losses) / len(losses)

def train_model(config: PretrainedConfig):
    run = wandb.init(project=config.project_name, name=datetime.now().strftime("%Y%m%d_%H%M%S"), config=config.__dict__)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = torch.device(config.device)

    print("training device: ", config.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print("torch.set_float32_matmul_precision as high")
    else:
        torch.set_float32_matmul_precision("medium")
        print("torch.set_float32_matmul_precision as medium")

    train_data = np.memmap(config.train_path, dtype=np.uint16, mode='r')
    file_size = os.path.getsize(config.valid_path)
    aligned_size = file_size - (file_size % 2)
    valid_data = np.memmap(config.valid_path, dtype=np.uint16, mode='r', offset=0, shape=(aligned_size // 2,))  #临时处理，valid的大小好像不是2的倍数

    model = Transformer(config.vocab_size, config.context_len, config.d_model, config.num_layers, config.num_heads, config.d_ff, config.rope_theta)
    model = model.to(config.device)
    print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    tokenizer = Tokenizer.from_files(config.vocab_path, config.merges_path)

    if config.use_compile:
        print("Using compile mode...")
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=config.lr, betas=(config.beta1,config.beta2), eps=config.epsilon, weight_decay=config.weight_decay)

    start_time = time.time()
    for step in tqdm(range(1, config.total_steps+1)):
        lr = lr_cosine_schedule(step, config.lr, config.lr*0.05, config.warmup_steps, config.total_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr

        loss = train(step, train_data, model, optimizer, config)

        if step % config.log_freq == 0:
            grad_norm = torch.sqrt(sum(x* x for x in [p.grad.data.norm() for p in model.parameters() if p.requires_grad]))
            wandb.log({
            'train/loss': loss, 
            'train/grad_norm': grad_norm, 
            'train/lr': lr, 
            'train/wallclock_time': time.time() - start_time
            }, step=step)
            print(f"step = {step}, loss = {loss}, lr = {lr}, grad_norm = {grad_norm}")
        
        if step % config.eval_freq == 0:
            eval_loss = evaluate(valid_data, model, config)
            wandb.log({
            'val/loss': eval_loss,
            'val/wallclock_time': time.time() - start_time
            }, step=step)
            print(f"step = {step}, eval_loss = {eval_loss}")
        
        if step % config.checkpoint_freq == 0:
            save_checkpoint(model, optimizer, step, os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt"))
            print(f"Checkpoint saved to {config.checkpoint_dir}/checkpoint_{step}.pt")

    eval_loss = evaluate(valid_data, model, config)
    wandb.log({
        'val/loss': eval_loss,
        'val/wallclock_time': time.time() - start_time
    }, step=step)
    print(f"final evaluation loss: {eval_loss}")

    save_checkpoint(model, optimizer, step, os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt"))

    wandb.finish()



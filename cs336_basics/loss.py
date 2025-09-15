import torch
import torch.nn as nn
from torch import Tensor
import math
from jaxtyping import Bool, Float, Int

def cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    original_shape = inputs.shape
    vocab_size = inputs.shape[-1]
    inputs = inputs.view(-1, vocab_size)
    targets = targets.view(-1)

    #数值稳定性
    logits_max = torch.max(inputs, dim=-1, keepdim=True).values #[batch_size, 1]
    logits_stable = inputs - logits_max #[batch_size, vocab_size]

    # log_softmax_i =  log(exp(x_i) / sum(exp_j))
    log_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True) #[batch_size, 1]
    log_softmax = logits_stable - log_sum_exp #[batch_size, vocab_size]

    batch_size = inputs.size(0)
    # 索引操作:
    #   torch.arange(log_softmax.shape[0]): [batch_size] (如 [0, 1, 2, ..., batch_size-1])
    #   targets: [batch_size]
    # 这相当于为每个batch样本选择对应的target索引位置
    # 输出形状: [batch_size]
    # selected_log_probs = torch.stack([
        # log_softmax[0, targets[0]],  # 第0个样本，选择targets[0]对应的概率
        # log_softmax[1, targets[1]],  # 第1个样本，选择targets[1]对应的概率
        # log_softmax[2, targets[2]],  # 第2个样本，选择targets[2]对应的概率
        # log_softmax[3, targets[3]]   # 第3个样本，选择targets[3]对应的概率
    # ])
    selected_log_probs = log_softmax[torch.arange(batch_size), targets]
    loss = -selected_log_probs.mean()
    return loss

    

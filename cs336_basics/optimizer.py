import torch
import torch.nn as nn
import math
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional, Dict, Any, List, Tuple

# optimizer = AdamW(model.parameters())

# # 方式1：常规使用（手动计算梯度）
# loss = criterion(model(inputs), targets)
# loss.backward()
# optimizer.step()

# # 方式2：使用closure（优化器内部重新计算）
# def closure():
#     optimizer.zero_grad()
#     output = model(inputs)
#     loss = criterion(output, targets)
#     loss.backward()
#     return loss

# optimizer.step(closure)  # 优化器内部会调用closure()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m = state['m']
                v = state['v']
                g = p.grad
                step = state['step']

                step += 1
                state['step'] = step

                m = beta1 * m + (1-beta1) * g
                v = beta2 * v + (1-beta2) * g * g

                state['m'] = m
                state['v'] = v

                alpha_t = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)

                p.data = p.data - alpha_t * m / (v.sqrt() + eps)
                if weight_decay > 0:
                    p.data = p.data - lr * weight_decay * p.data

        return loss
        
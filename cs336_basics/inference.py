import torch
from transformers import AutoTokenizer
from .trainer import load_checkpoint
from .model import Softmax, Transformer

def generate_text(model: Transformer, tokenizer: AutoTokenizer, prompt: str, new_tokens_limit: int=32, temperature: float=0.5, top_p: float=0.9):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(tokenizer.encode(prompt), device=device)
    x = x.unsqueeze(0)
    context_length = model.context_length
    for _ in range(new_tokens_limit):
        x = x[:, -context_length:] if x.size(1) > context_length else x
        logits = model(x)
        # next_token_logits：[1, vocab_size]
        next_token_logits = logits[:, -1, :] 
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
        #sorted_logits, sorted_indices：[1, vocab_size]，分别代表排序后的logits和对应的原始索引
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True) 
        softmax_ = Softmax()
        next_token_prob = softmax_(sorted_logits, dim=-1)
        #计算累积概率，[1, vocab_size]
        cum_probs = torch.cumsum(next_token_prob, dim=-1)
        mask = (cum_probs <= top_p)
        mask[..., 0] = True #至少有一个token被选中
        # filtered_probs：[1, vocab_size]
        filtered_probs = torch.where(mask, cum_probs, torch.zeros_like(cum_probs))
        #归一化
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
        #torch.multinomial() 的作用是从给定的概率分布中进行多项式采样，即根据每个元素的概率权重随机选择一个索引
        # sampled_indices：[1, 1]
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
        # torch.gather(input, dim, index, *, sparse_grad=False, out=None)
        # input: 源张量，从中收集数据
        # dim: 收集操作的维度
        # index: 索引张量，指定要收集的元素位置
        # sparse_grad: 是否使用稀疏梯度（默认为 False）
        # out: 输出张量（可选）
        # next_token_idx: [1, 1]
        next_token_idx = sorted_indices.gather(-1, sampled_indices)

        if next_token_idx[0, 0] == tokenizer.eos_token_id:
            break
        # x: [1, seq_len+1]
        x = torch.cat((x, next_token_idx), dim=-1)
    return tokenizer.decode(x.squeeze(0).tolist())

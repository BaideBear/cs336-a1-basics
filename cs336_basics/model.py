import torch
import torch.nn as nn
import math
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, weights=None, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        if weights is None:
            weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
            std = math.sqrt(2.0 / (in_features + out_features))
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0*std, b = 3.0*std)
            self.W = nn.Parameter(weight)
        else:
            self.W = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.t()
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weights=None, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        if weights is None:
            weights = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
            nn.init.trunc_normal_(weights, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.E = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.E[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, weights=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        if weights is None:
            weights = torch.ones(d_model)
        self.G = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        res = x/rms_x * self.G
        return res.to(in_dtype)
    
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, w1_weight: None, w2_weight: None, w3_weight: None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, w1_weight)
        self.w2 = Linear(d_ff, d_model, w2_weight)
        self.w3 = Linear(d_model, d_ff, w3_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        glu_x = (self.w1(x) * torch.sigmoid(self.w1(x))) * self.w3(x)
        return self.w2(glu_x)

class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        assert d_k % 2 == 0

        # freqs_k = 1/(theta^(2k/d_k)), k = 0, 1, ... , d_k/2 - 1
        # freqs: [d_k//2]
        freqs = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        # pos: [max_seq_len, ] 
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        # 外积，angles: [max_seq_len, d_k//2]
        # theta_{i,k} = angles[i,k]
        angles = torch.outer(pos, freqs)

        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        # persistent=False表示这些缓冲区不会保存到模型状态字典中
        # sin_cached, cos_cached: [max_seq_len, d_k//2]
        self.register_buffer("cos_cached", cos_values, persistent=False)
        self.register_buffer("sin_cached", sin_values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: [..., seq_len, d_k]
        token_positions: [..., seq_len]
        """
        #将x和token_positions在seq_len前的形状广播到一致
        if token_positions.dim()+1 < x.dim():
            t = x.dim()-token_positions.dim()-1
            for _ in range(t):
                token_positions = token_positions.unsqueeze(0)
        # print(f"x shape: {x.shape}")
        # print(f"token_pos shape:{token_positions.shape}")
        # x_shaped: [..., seq_len, d_k//2, 2]
        x_shaped = rearrange(x, "... seq_len (d_pair pair) -> ... seq_len d_pair pair", pair=2)

        # expand_pos: [..., seq_len], 不需要expand操作，实际上索引操作不会消除最后一列
        expand_pos = token_positions
        # torch的索引功能，用cos_cached的第一列作为索引，得到expand_pos最后一列的查询结果
        # cos: [..., seq_len, d_k//2]
        cos = self.cos_cached[expand_pos]
        sin = self.sin_cached[expand_pos]

        # x1, x2: [..., seq_len, d_k//2]
        x1 = x_shaped[..., 0]
        x2 = x_shaped[..., 1]

        # print(f"x1 shape: {x1.shape}")
        # print(f"cos shape: {cos.shape}")

        rotated_x1 = x1*cos - x2*sin
        rotated_x2 = x1*sin + x2*cos

        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)

        res = rearrange(rotated_x, "... seq_len d_pair pair -> ... seq_len (d_pair pair)")

        return res
    
class Softmax(nn.Module):
    def forward(self, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        #对于torch.max：values代表最大值张量，indices代表索引张量
        max_vals = torch.max(x, dim=dim, keepdim=True).values
        #数值稳定化处理
        x_stable = x - max_vals
        exp_x = torch.exp(x_stable)
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        return exp_x / sum_exp
    
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    O = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        masked_O = O.masked_fill(mask==False, float('-inf'))
    else:
        masked_O = O
    softmax_ = Softmax()
    return softmax_(masked_O) @ V

class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None=None, 
                    use_rope: bool=False, theta: float | None = None, 
                    q_proj_weight: Float[Tensor, " d_k d_model"]=None,
                    k_proj_weight: Float[Tensor, " d_k d_model"]=None,
                    v_proj_weight: Float[Tensor, " d_v d_model"]=None,
                    o_proj_weight: Float[Tensor, " d_model d_v"]=None,
                    token_positions: Int[Tensor, " ... sequence_length"] | None = None) :
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, q_proj_weight)
        self.k_proj = Linear(d_model, d_model, k_proj_weight)
        self.v_proj = Linear(d_model, d_model, v_proj_weight)
        self.o_proj = Linear(d_model, d_model, o_proj_weight)

        self.use_rope = use_rope
        self.rope = RoPE(self.d_k, theta, max_seq_len) if use_rope else None
        self.token_positions = token_positions

    def forward(self, x: Float[Tensor, "... seq_len d_model"]):
        seq_len = x.size(-2)
        qkv_proj = torch.cat([self.q_proj.W, self.k_proj.W, self.v_proj.W], dim=0)
        qkv = x @ qkv_proj.t()
        q, k, v = qkv.chunk(3, -1) #d_k = d_v
        
        q = rearrange(q, "... seq (h hd) -> ... h seq hd", h=self.num_heads)
        k = rearrange(k, "... seq (h hd) -> ... h seq hd", h=self.num_heads)
        v = rearrange(v, "... seq (h hd) -> ... h seq hd", h=self.num_heads)

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)

        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, "... h seq hd -> ... seq (h hd)")
        return self.o_proj(output)

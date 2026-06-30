"""Shared neural network layers used by PBSQ, ViT backbones, and comparison modules."""

from .common import *

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Residual_droppath(nn.Module):
    def __init__(self, fn,drop_path_rate=0.1):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    def forward(self, x, **kwargs):
        return self.drop_path(self.fn(x, **kwargs)) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.,last_dim=None):
        super().__init__()
        if last_dim==None:
            last_dim=dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, last_dim),
            nn.Dropout(dropout)
        )
        # else:
            
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        # self.to_qkv = nn.Linear(dim, np.int64(inner_dim/2) , bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(np.int64(inner_dim/6), dim),
        #     nn.Dropout(dropout)
        # )
        self.attention_score = 0

    def forward(self, x, mask = None):
        # pdb.set_trace()
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        # pdb.set_trace()
        self.attention_score=attn.detach()
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

class Transformer(nn.Module):
    # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
    #     super().__init__()
    #     self.layers = nn.ModuleList([])
    #     for _ in range(depth):
    #         self.layers.append(nn.ModuleList([
    #             Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
    #             Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
    #         ]))
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout,last_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        if last_dim==None:
            last_dim=dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_droppath(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual_droppath(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        # pdb.set_trace()
        # self.layers.append(nn.ModuleList([
        #         Residual_droppath(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
        #         Residual_droppath(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout,last_dim=last_dim)))
        #     ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            #embed()
            x = ff(x)
        return x

def MLP(channels: list, do_bn=True):
    # Multi-layer perceptron
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(f"activation should be relu/gelu/glu/leaky_relu, not {activation}.")

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, q, k, v):
        if k.dim() == 3:
            attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        else:  
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn, torch.log(attn + 1e-8)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, with_W=False):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.with_W = with_W
        
        if with_W:
            self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        else:
            self.linears = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # Apply linear transformations and reshape
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        # Apply attention
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # Concatenate heads and apply final linear transformation
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class FFN_MLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(FFN_MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SingleHeadSiameseAttention(nn.Module):
   
    def __init__(self, d_model):
        super().__init__()
        self.n_head = 1
        self.d_model = d_model
        self.w_qk = nn.Linear(self.d_model, self.n_head * self.d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_model, 0.5))
        nn.init.normal_(self.w_qk.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_model)))

        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)

        self.linear1 = nn.Sequential(nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)

    def forward(self, q, k, v, tsp):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = v.view(sz_b, len_v, self.n_head, self.d_model)

        tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head, self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)
        tsp = torch.cat([tsp, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_model)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1, self.d_model)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1, self.d_model)
        tsp = tsp.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1, self.d_model)

        output, attn, log_attn = self.attention(q, k, v)
        tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        tsp = tsp.view(self.n_head, sz_b, len_q, self.d_model)
        tsp = tsp.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output1 = self.linear1(output * residual)
        output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )

        return output, tsp

class FFN_MLP_one(nn.Module):
    def __init__(self, feature_dim, scale=1.0):
        super(FFN_MLP_one, self).__init__()
        self.linear1 = nn.Linear(feature_dim, feature_dim)

    def forward(self, src):
        src = scale * self.linear1(src) + src # (B, h*w, c)
        return src

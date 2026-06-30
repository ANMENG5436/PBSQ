"""Prior-Aware Query (PAQ), Prior-Driven Activation (PDA), and PBSQ modules."""

from .common import *
from .layers import FFN_MLP, ScaledDotProductAttention

class PBSQ_EASY(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP'):
        super(PBSQ_EASY, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.FFN_method = FFN_method
        # query embedding
        
       
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        self.gate_proj = nn.Linear(self.feature_dim, self.num_queries)
        self.query_modulator = nn.Linear(self.feature_dim, self.feature_dim)
        
        # attention
        self.with_W = False
        self.attention = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5))
            
        # FFN
        if self.FFN_method == 'MLP':
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
            
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, src):
      
        if src.dim() == 3:
            B, seq_len, dim = src.size()
            cls_token = src[:, 0:1, :] 
            patch_tokens = src[:, 1:, :]  
            
         
            cls_feature = cls_token.expand_as(patch_tokens)  # [B,196,D]
            patch_tokens = patch_tokens + cls_feature 
            
            
            h = w = int(math.sqrt(seq_len - 1))
            src = patch_tokens.permute(0,2,1).view(B,dim,h,w)
            has_cls_token = True
        else:
            has_cls_token = False
        
        B, c, h, w = src.size()
        assert c == self.feature_dim
        src = src.contiguous().view(B, c, h*w).transpose(1, 2)  # (B, h*w, c)
        
        
        img_features = src  # (B, h*w, c)
        query_embed_weight = self.query_embed.weight.unsqueeze(0).repeat(B,1,1)  # (B, num_queries, c)

       
        output1, _, softmax_qk1 = self.attention(img_features, query_embed_weight, query_embed_weight)  # (B, h*w, c)
       
        output2, _, softmax_qk2 = self.attention(query_embed_weight, img_features, img_features)  # (B, num_queries, c)
       
        output2 = self.attention(img_features, output2, output2)[0]  # (B, h*w, c)
            
        output =  output1+output2  # (B, h*w, c)
        
           
        softmax_qk = None 
            
        
        if self.FFN_method == 'MLP':
            output = self.FFN(output)  # (B, h*w, c)
        
        
        output = output.transpose(1, 2).contiguous().view(B, c, h, w)  # (B, c, h, w)
        
      
        if has_cls_token:
            output = output.view(B, self.feature_dim, h*w).permute(0,2,1)  # [B,196,D]
            output = torch.cat([cls_token, output], dim=1)  # [B,197,D]
        
        return output, softmax_qk

class PAQ(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, dropout=0.1, 
                 enable_dynamic_modulation=False, attention=None):
        super(PAQ, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.enable_dynamic_modulation = enable_dynamic_modulation
        
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        
        # 门控调制网络 M(·) 
        self.modulation_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # 门控函数 G(·) 
        self.gate_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim),
            nn.Sigmoid()
        )
        
        # 接收外部传入的共享attention模块
        if attention is None:
            # 如果没有传入,则自己创建(用于独立使用PAQ的情况)
            self.attention = ScaledDotProductAttention(
                temperature=np.power(self.feature_dim, 0.5)
            )
        else:
            self.attention = attention
        
        if self.enable_dynamic_modulation:
            self.layer_norm = nn.LayerNorm(self.feature_dim)
    
    def forward(self, img_features):
        B, L, C = img_features.size()
        
        # Step 1: 获取基础查询嵌入
        base_queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, C]
        
        # Step 2: 根据开关决定是否应用动态调制
        if self.enable_dynamic_modulation:
            f_bar = torch.mean(img_features, dim=1)  # [B, C]
            modulation_vector = self.modulation_net(f_bar)  # [B, C]
            gate_weights = self.gate_net(f_bar)  # [B, C]
            
            modulation_expanded = modulation_vector.unsqueeze(1).repeat(1, self.num_queries, 1)
            gate_expanded = gate_weights.unsqueeze(1).repeat(1, self.num_queries, 1)
            
            dynamic_queries = base_queries + gate_expanded * modulation_expanded
            dynamic_queries = self.layer_norm(dynamic_queries)
        else:
            dynamic_queries = base_queries
        
        # Step 3: 执行注意力机制 - attention(img_features, queries, queries)
        perceived_features, _, attention_weights = self.attention(
            q=img_features,      # 查询：图像特征
            k=dynamic_queries,   # 键：动态查询嵌入  
            v=dynamic_queries    # 值：动态查询嵌入
        )
        
        return perceived_features, dynamic_queries, attention_weights

class PDA(nn.Module):
    def __init__(self, feature_dim, num_queries=100, FFN_method='MLP', attention=None):
        super(PDA, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.FFN_method = FFN_method
        
        # 接收外部传入的共享attention模块
        if attention is None:
            self.attention = ScaledDotProductAttention(
                temperature=np.power(self.feature_dim, 0.5)
            )
        else:
            self.attention = attention
        
        if self.FFN_method == 'MLP':
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
    
    def forward(self, img_features, queries):
        # attention(queries, img_features, img_features)
        driven_queries, _, attention_weights = self.attention(
            queries, img_features, img_features
        )
        
        # attention(img_features, driven_queries, driven_queries)
        activated_features = self.attention(img_features, driven_queries, driven_queries)[0]
        
        return activated_features, attention_weights

class PBSQ(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP',
                 enable_pda=True, paq_weight=1.0, pda_weight=1.0, fusion_method='add',
                 enable_dynamic_modulation=False, dropout=0.1):
        super(PBSQ, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.FFN_method = FFN_method
        self.enable_pda = enable_pda
        self.paq_weight = paq_weight
        self.pda_weight = pda_weight
        self.fusion_method = fusion_method
        
        # 创建共享的attention模块
        self.shared_attention = ScaledDotProductAttention(
            temperature=np.power(self.feature_dim, 0.5)
        )
        
        # 将共享attention传入PAQ和PDA
        self.PAQ = PAQ(feature_dim, num_queries, num_heads, dropout, 
                       enable_dynamic_modulation, attention=self.shared_attention)
        
        if self.enable_pda:
            self.PDA = PDA(feature_dim, num_queries, FFN_method, 
                          attention=self.shared_attention)
        
        if self.FFN_method == 'MLP':
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
        
        # 根据融合方法添加相应组件
        if self.fusion_method == 'gate' and self.enable_pda:
            self.fusion_gate = nn.Sequential(
                nn.Linear(2*self.feature_dim, self.feature_dim),
                nn.ReLU(),
                nn.Linear(self.feature_dim, 2),
                nn.Softmax(dim=-1)
            )
        elif self.fusion_method == 'weighted' and self.enable_pda:
            self.learnable_paq_weight = nn.Parameter(torch.tensor(self.paq_weight))
            self.learnable_pda_weight = nn.Parameter(torch.tensor(self.pda_weight))
    
    def forward(self, src):
        if src.dim() == 3:
            B, seq_len, dim = src.size()
            cls_token = src[:, 0:1, :]
            patch_tokens = src[:, 1:, :]
            
            cls_feature = cls_token.expand_as(patch_tokens)
            patch_tokens = patch_tokens + cls_feature
            
            h = w = int(math.sqrt(seq_len - 1))
            src = patch_tokens.permute(0,2,1).view(B,dim,h,w)
            has_cls_token = True
        else:
            has_cls_token = False
        
        B, c, h, w = src.size()
        assert c == self.feature_dim
        src = src.contiguous().view(B, c, h*w).transpose(1, 2)  # (B, h*w, c)
        
        img_features = src
        
        # PAQ: 先验感知查询
        # 内部调用: attention(img_features, queries, queries)
        perceived_features, queries, attention_weights_paq = self.PAQ(img_features)
        
        # 根据enable_pda决定是否执行PDA
        if self.enable_pda:
            # PDA: 先验驱动激活
            # 内部调用: attention(queries, img_features, img_features)
            #          attention(img_features, output2, output2)
            activated_features, attention_weights_pda = self.PDA(img_features, queries)
            
            # 根据融合方法合并特征
            if self.fusion_method == 'add':
                output = self.paq_weight * perceived_features + self.pda_weight * activated_features
            elif self.fusion_method == 'gate':
                concat_features = torch.cat([perceived_features, activated_features], dim=-1)
                gates = self.fusion_gate(concat_features)
                gate1, gate2 = gates[..., 0:1], gates[..., 1:2]
                output = gate1 * perceived_features + gate2 * activated_features
            elif self.fusion_method == 'weighted':
                output = (self.learnable_paq_weight * perceived_features + 
                         self.learnable_pda_weight * activated_features)
            else:
                raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")
            
            attention_weights = attention_weights_pda
        else:
            output = self.paq_weight * perceived_features
            attention_weights = attention_weights_paq
        
        # FFN处理
        if self.FFN_method == 'MLP':
            output = self.FFN(output)
        
        # 转回原始形状
        output = output.transpose(1, 2).contiguous().view(B, c, h, w)
        
        if has_cls_token:
            output = output.view(B, self.feature_dim, h*w).permute(0,2,1)
            output = torch.cat([cls_token, output], dim=1)
        
        return output, attention_weights
    
    @classmethod
    def create_ablation_variant(cls, variant_name, **kwargs):
        """工厂方法创建不同消融实验变体"""
        configs = {
            'baseline': {'enable_pda': False, 'fusion_method': 'add'},
            'paq_static': {'enable_pda': False, 'enable_dynamic_modulation': False},
            'paq_dynamic': {'enable_pda': False, 'enable_dynamic_modulation': True},
            'pbsq_add': {'enable_pda': True, 'fusion_method': 'add'},
            'pbsq_gate': {'enable_pda': True, 'fusion_method': 'gate'},
        }
        config = configs.get(variant_name, {})
        config.update(kwargs)
        return cls(**config)

def power_transform(features: Tensor, power_factor: float = 0.5) -> Tensor:
    """
    Apply power transform to features for better few-shot learning performance.
    
    Args:
        features: Input features of shape (..., feature_dim)
        power_factor: Power factor for transformation (default: 0.5 for square root)
    
    Returns:
        Power transformed features with same shape as input
    """
    # Ensure features are non-negative by taking absolute value
    features_abs = torch.abs(features)
    
    # Apply power transform: sign(x) * |x|^power_factor
    # This preserves the sign while applying the power transformation
    transformed = torch.sign(features) * torch.pow(features_abs + 1e-12, power_factor)
    
    return transformed

class AdaptivePrototypeRefinement(nn.Module):
    """Adaptive prototype refinement module with learnable parameters"""
    def __init__(self, feature_dim, temperature=10.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.prototype_adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, prototypes, features):
        # Adapt prototypes based on current features
        adapted_prototypes = prototypes + self.prototype_adapter(prototypes)
        
        # Compute adaptive similarities
        similarities = F.cosine_similarity(
            features.unsqueeze(1), 
            adapted_prototypes.unsqueeze(0), 
            dim=-1
        ) * self.temperature
        
        return adapted_prototypes, similarities


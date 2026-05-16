import torch
import numpy as np
from torch import nn
from TFEngine import TFEngine


# 连续傅里叶时间嵌入

class LearnableFourierEmbedding(nn.Module):
    def __init__(self, num_time_steps, rank_c, num_freqs=16):
        super(LearnableFourierEmbedding, self).__init__()
        #生成num_freqs个从0.1到3.14均匀分布的数值，作为可学习参数
        #0.1代表长周期，3.14代表短周期
        #omega_min=2*3.1415926 / num_time_steps
        freqs = torch.linspace(0.1, 3.14, num_freqs)
        #激活频率，反向传播
        self.frequencies = nn.Parameter(freqs)
        #数据=周期规律+突发噪声
        #提取傅里叶正余弦波，映射到rank_c（时间维度的大小），提取周期规律
        self.spectral_linear = nn.Linear(num_freqs * 2, rank_c)
        #传统embedding拟合突发噪声
        #num_time_step来自train.py 中：max_C_index = torch.max(train_dataset.C_Indices) + 1，config["dimensions"][2]，
        #也就是 self.max_cID。最终，在初始化傅里叶模块时，self.max_cID 作为参数传了进来
        self.residual_embedding = nn.Embedding(num_time_steps, rank_c)
        

        #线性层xavier初始化
        nn.init.xavier_uniform_(self.spectral_linear.weight)
        #离散嵌入控制在-0.01到0.01，影响较小，根据不同场景可适当调整
        nn.init.uniform_(self.residual_embedding.weight, -0.01, 0.01)

    def forward(self, t_idx):

        #计算相位wt，与离散的t=1,t=2不同，可以提取规律，能够明白今天的现在和昨天的现在都在同一个相位
        #t_idx为时间索引（int），来自NeuTucF代码中的cIDs，转换为浮点数，升维
        t = t_idx.float().unsqueeze(-1)
        wt = t * self.frequencies 
        #提取傅里叶特征，计算正弦和余弦并拼接，16个频率变成32维特征（在第二个维度拼接）
        fourier_features = torch.cat([torch.sin(wt), torch.cos(wt)], dim=-1)
        #周期流形
        c_spectral = self.spectral_linear(fourier_features)
        #突发残差
        c_residual = self.residual_embedding(t_idx)
        #c_emb=c_spectral + c_residual，时间嵌入
        return c_spectral + c_residual, fourier_features


# 时空联合动态门控 (Spatio-Temporal Co-Gating)

class SpatioTemporalGating(nn.Module):
    def __init__(self, num_freqs, rank_a, rank_b, interaction_dim):
        super(SpatioTemporalGating, self).__init__()
        # 输入维度 = 频域特征维度 + 空间A的维度 + 空间B的维度
        in_features = (num_freqs * 2) + rank_a + rank_b
        
        # 将时空联合特征映射到 125 维的 Tucker 交互空间，生成个性化 Gate
        self.st_to_gate = nn.Sequential(
            nn.Linear(in_features, interaction_dim),
            nn.Sigmoid() 
        )

        #Gate数值容器
        self.saved_gates=None

    def forward(self, interactions, fourier_features, a_emb, b_emb):
        # 拼接时间频率、空间 A 和空间 B 的特征
        # st_features 维度:[batch_size, num_freqs*2 + rank_a + rank_b]
        st_features = torch.cat([fourier_features, a_emb, b_emb], dim=-1)
        
        # 生成包含空间信息的时间门控，吐出一个权重
        # dynamic_gate 维度: [batch_size, interaction_dim]
        dynamic_gate = self.st_to_gate(st_features)

        #保存gate mask
        if not self.training:
            self.saved_gates=dynamic_gate.detach().cpu().numpy()
        
        # Hadamard product
        return interactions * dynamic_gate


# 主模型：SANeuTF

class SANeuTF(nn.Module):
    def __init__(self, config):
        super(SANeuTF, self).__init__()
        self.config = config
        self.max_aID = config["dimensions"][0]
        self.max_bID = config["dimensions"][1]
        self.max_cID = config["dimensions"][2]
        ranks = config["embedding_dims"]
        self.flag_bias = config["flag_bias"]

        if isinstance(ranks, int):
            self.rankA = self.rankB = self.rankC = ranks
        else:
            self.rankA, self.rankB, self.rankC = ranks[0], ranks[1], ranks[2]

        self.interaction_dim = self.rankA * self.rankB * self.rankC

        # 空间 Embedding (离散)
        self.embedding_layer_A = nn.Embedding(self.max_aID, self.rankA)
        self.embedding_layer_B = nn.Embedding(self.max_bID, self.rankB)
        
        # 时间 Embedding (连续傅里叶)
        self.num_freqs = 64
        self.embedding_layer_C = LearnableFourierEmbedding(self.max_cID, self.rankC, self.num_freqs)

        # 时空联合门控
        self.st_gating = SpatioTemporalGating(
            num_freqs=self.num_freqs, 
            rank_a=self.rankA, 
            rank_b=self.rankB, 
            interaction_dim=self.interaction_dim
        )

        #  NeuTucF 核心张量层
        self.tucker_linear_layer = nn.Linear(self.interaction_dim, 1, bias=self.flag_bias)
        self.output_mapping = nn.Sigmoid()

        # 初始化
        nn.init.uniform_(self.embedding_layer_A.weight, 0, 1)
        nn.init.uniform_(self.embedding_layer_B.weight, 0, 1)
        nn.init.uniform_(self.tucker_linear_layer.weight, 0, 1)
        if self.flag_bias:
            nn.init.constant_(self.tucker_linear_layer.bias, 0)

        nn.init.xavier_normal_(self.st_gating.st_to_gate[0].weight, gain=0.1)
        nn.init.constant_(self.st_gating.st_to_gate[0].bias, 0.0)

    def forward(self, aIDs, bIDs, cIDs):
        device = next(self.parameters()).device
        aIDs, bIDs, cIDs = aIDs.to(device), bIDs.to(device), cIDs.to(device)

        # 获取特征
        a_emb = self.embedding_layer_A(aIDs)
        b_emb = self.embedding_layer_B(bIDs)
        c_emb, fourier_features = self.embedding_layer_C(cIDs)

        # Tucker 外积
        outer_product = torch.einsum("bi,bj,bk->bijk", a_emb, b_emb, c_emb)
        #展平
        interactions = outer_product.view(-1, self.interaction_dim)

        # 时空联合门控 (Spatio-Temporal Co-Gating)
        # 将 A, B 的空间位置特征传递给门控
        gated_interactions = self.st_gating(interactions, fourier_features, a_emb, b_emb)

        # 线性打分与输出
        logits = self.tucker_linear_layer(gated_interactions)
        out = self.output_mapping(logits)
        
        if not self.training:
            out = out.cpu()
        return out

    def load_model(self, model_dir: str):
        device = next(self.parameters()).device
        state_dict = torch.load(model_dir, map_location=device)
        self.load_state_dict(state_dict)


# 训练引擎

class SANeuTFEngine(TFEngine):
    def __init__(self, config, bounds=[0, 1], flag_norm=False, flag_log=False):
        self.model = SANeuTF(config)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif config.get("use_mps", False) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        super(SANeuTFEngine, self).__init__(config, bounds, flag_norm, flag_log)

    def train_single_batch(self, users, services, timeslices, ratings):
        ratings = ratings.to(self.device)
        return super().train_single_batch(users, services, timeslices, ratings)
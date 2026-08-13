import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import utils


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class CoLaKG(BasicModel):
    # 在 LightGCN 基础上，引入：
    # 1. 物品/用户的 LLM 语义 Embedding（1024维）
    # 2. 基于语义相似度的 GAT 邻居注意力聚合
    # 支持消融实验：
    #   ablation=1：移除物品侧语义向量（物品融合与邻居注意力均使用协同Embedding）
    #   ablation=2：移除邻居注意力，改为均值聚合
    #   ablation=3：去掉注意力拼接中的用户-邻居逐元素乘积（Wu*Wh）
    #   ablation=4：移除用户侧语义向量（用户融合与邻居注意力均使用协同Embedding）
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset,
                 adj_matrix=None,         # top-k 语义邻居索引，shape: (m_items, k)
                 semantic_emb=None,       # 物品语义 Embedding，shape: (m_items, 1024)
                 user_semantic_emb=None,  # 用户语义 Embedding，shape: (n_users, 1024)
                 ):
        super(CoLaKG, self).__init__()
        self.config = config
        self.dataset : BasicDataset = dataset
        self.adj_matrix = adj_matrix.to(world.device)
        self.semantic_emb = semantic_emb.to(world.device)
        self.user_semantic_emb = user_semantic_emb.to(world.device)
        self.dropout_i = self.config['dropout_i']
        self.dropout_u = self.config['dropout_u']
        self.dropout_neighbor = self.config['dropout_n']
        self.ablation = self.config.get('ablation', 0)
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("self.num_items", self.num_items)
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers   = self.config['lightGCN_n_layers']
        self.keep_prob  = self.config['keep_prob']

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        # 语义映射层：将 1024 维语义向量压缩到 latent_dim（64）维
        # ablation=1 不使用物品语义，无需 semantic_map
        if self.ablation != 1:
            self.semantic_map = nn.Linear(1024, self.latent_dim)
        # ablation=4 不使用用户语义，无需 user_semantic_map
        if self.ablation != 4:
            self.user_semantic_map = nn.Linear(1024, self.latent_dim)
        print(f"lgn is already to go(drop_edge:{self.config['use_drop_edge']}), ablation={self.ablation}")

        # GAT 注意力参数（消融2：均值聚合，不需要任何注意力参数）
        if self.ablation != 2:
            # ablation=1：邻居注意力改用协同 Embedding（latent_dim），否则用语义 Embedding（1024）
            emb_dim = self.latent_dim if self.ablation == 1 else 1024
            self.W = nn.Parameter(torch.empty(size=(emb_dim, 32)))
            nn.init.xavier_uniform_(self.W.data, gain=1.414)

            # ablation=3：去掉 Wu*Wh，拼接维度从 3*32=96 缩减到 2*32=64
            attn_concat_dim = 2 * 32 if self.ablation == 3 else 3 * 32
            self.a = nn.Parameter(torch.empty(size=(attn_concat_dim, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

            # ablation=3 不需要用户投影矩阵（Wu*Wh 项已去掉）
            if self.ablation != 3:
                self.W_u_attn = nn.Parameter(torch.empty(size=(self.latent_dim, 32)))
                nn.init.xavier_uniform_(self.W_u_attn.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        return self.__dropout_x(self.Graph, keep_prob)

    def computer(self):
        # CoLaKG 核心传播，分三个阶段：
        # 1. 语义融合：协同 Embedding + 语义 Embedding（等权平均）
        # 2. 用户感知 GAT 邻居聚合（含三种消融变体）
        # 3. 图卷积传播：在用户-物品图上做 LightGCN

        # === 阶段1：用户语义融合 ===
        # ablation=4：移除用户侧语义向量，仅使用协同 Embedding
        if self.ablation == 4:
            users_emb_merged = self.embedding_user.weight
            user_context = self.embedding_user.weight.mean(0, keepdim=True)  # [1, latent_dim]
        else:
            user_semantic_emb = F.dropout(self.user_semantic_emb, self.dropout_u, training=self.training)
            user_semantic_emb = self.user_semantic_map(user_semantic_emb)  # 1024 → 64
            user_semantic_emb = F.elu(user_semantic_emb)
            user_semantic_emb = F.dropout(user_semantic_emb, self.dropout_u, training=self.training)
            users_emb_merged = (self.embedding_user.weight + user_semantic_emb) / 2
            user_context = user_semantic_emb.mean(0, keepdim=True)  # [1, latent_dim]

        # === 阶段1：物品语义融合 ===
        # ablation=1：移除物品侧语义向量，仅使用协同 Embedding
        if self.ablation == 1:
            items_emb_merged = self.embedding_item.weight
        else:
            items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
            items_semantic_emb = self.semantic_map(items_semantic_emb)  # 1024 → 64
            items_semantic_emb = F.elu(items_semantic_emb)
            items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
            items_emb_merged = (self.embedding_item.weight + items_semantic_emb) / 2

        # === 阶段2：邻居聚合 ===
        neighbor_emb = items_emb_merged[self.adj_matrix]  # (N, k, latent_dim)

        if self.ablation == 2:
            # ablation=2：移除注意力计算，改为均值权重
            h_prime = neighbor_emb.mean(dim=1)  # (N, latent_dim)
            h_prime = F.elu(h_prime)
        else:
            k = self.adj_matrix.shape[1]
            if self.ablation == 1:
                # ablation=1：用协同 Embedding 计算注意力权重。
                # detach() 使梯度不经注意力路径回传到 embedding_item.weight，
                # 与原模型保持一致（原模型中 semantic_emb 无梯度）。
                # 同时复用 neighbor_emb 避免重复索引，先投影再 expand 减少中间张量。
                Wh  = torch.matmul(neighbor_emb.detach(), self.W)              # (N, k, 32)
                self_proj = torch.matmul(items_emb_merged.detach(), self.W)    # (N, 32)
                Wh0 = self_proj.unsqueeze(1).expand(-1, k, -1)                 # (N, k, 32)
            else:
                neighbor_repr = self.semantic_emb[self.adj_matrix]             # (N, k, 1024)
                self_proj = torch.matmul(self.semantic_emb, self.W)            # (N, 32)
                Wh  = torch.matmul(neighbor_repr, self.W)                      # (N, k, 32)
                Wh0 = self_proj.unsqueeze(1).expand(-1, k, -1)                 # (N, k, 32)

            if self.ablation == 3:
                # ablation=3：去掉用户-邻居逐元素乘积（Wu*Wh），仅拼接 [Wh0 || Wh]
                W_concat = torch.cat((Wh0, Wh), dim=-1)                           # (N, k, 64)
            else:
                Wu = torch.matmul(user_context, self.W_u_attn)                    # (1, 32)
                Wu = Wu.unsqueeze(0).expand(neighbor_emb.shape[0], k, -1)         # (N, k, 32)
                Wu_Wh = Wu * Wh                                                    # (N, k, 32)
                W_concat = torch.cat((Wh0, Wh, Wu_Wh), dim=-1)                    # (N, k, 96)

            attention = torch.matmul(W_concat, self.a).squeeze(-1)                # (N, k)
            attention = self.leakyrelu(attention)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout_neighbor, training=self.training)
            attention = attention.unsqueeze(-1)                                    # (N, k, 1)

            h_prime = torch.sum(attention * neighbor_emb, dim=1)                  # (N, latent_dim)
            h_prime = F.elu(h_prime)

        items_emb_merged = (items_emb_merged + h_prime) / 2

        # === 阶段3：LightGCN 图卷积传播 ===
        all_emb = torch.cat([users_emb_merged, items_emb_merged])
        embs = [all_emb]

        if self.config['use_drop_edge'] and self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        light_out = torch.mean(torch.stack(embs, dim=1), dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        rating = self.f(torch.matmul(users_emb, all_items.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb   = all_items[pos_items]
        neg_emb   = all_items[neg_items]

        # 原始协同 Embedding（用于正则化）
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego   = self.embedding_item(pos_items)
        neg_emb_ego   = self.embedding_item(neg_items)

        # 用户语义映射后的 Embedding（用于正则化）
        # ablation=4：不使用用户语义，对应正则化项置零
        if self.ablation == 4:
            users_emb_ego0 = torch.zeros_like(users_emb_ego)
        else:
            users_emb_ego0 = self.user_semantic_map(self.user_semantic_emb)[users]

        # ablation=1：不使用物品语义，对应正则化项置零
        if self.ablation == 1:
            pos_emb_ego0 = torch.zeros_like(pos_emb_ego)
            neg_emb_ego0 = torch.zeros_like(neg_emb_ego)
        else:
            pos_emb_ego0 = self.semantic_map(self.semantic_emb)[pos_items]
            neg_emb_ego0 = self.semantic_map(self.semantic_emb)[neg_items]

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, pos_emb_ego0, neg_emb_ego0, users_emb_ego0

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0,
         pos_emb_ego0, neg_emb_ego0, users_emb_ego0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        # L2 正则化：同时约束协同 Embedding 和语义映射 Embedding
        reg_loss = (1/2) * (userEmb0.norm(2).pow(2) +
                            posEmb0.norm(2).pow(2) +
                            negEmb0.norm(2).pow(2) +
                            pos_emb_ego0.norm(2).pow(2) +
                            neg_emb_ego0.norm(2).pow(2) +
                            users_emb_ego0.norm(2).pow(2)
                            ) / float(len(users))

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

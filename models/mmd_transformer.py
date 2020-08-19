import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omni_torch.networks.blocks import *
from omni_torch.networks.activation import *


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention for image'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, q, k, v, dist_func):
        q = q / self.temperature
        if dist_func.lower() == "matmul":
            attn = torch.matmul(q.permute(0, 2, 1), k)
        else:
            q = q.unsqueeze(2).repeat(1, 1, k.size(2), 1)
            k = k.unsqueeze(3).repeat(1, 1, 1, q.size(3))
            if dist_func.lower() in ["l1"]:
                attn = torch.mean(torch.abs(q - k), dim=1)
            elif dist_func.lower() in ["l2", "euclidean"]:
                attn = torch.mean(torch.sqrt((q - k) ** 2), dim=1)
            elif dist_func.lower() in ["cos", "cosine"]:
                attn = 1 - torch.acos(self.cos(q, k)) / math.pi
                # attn = self.cos(q, k)
            else:
                raise NotImplementedError()
        attn = F.softmax(attn, dim=-1)
        # apply attention on spatial location
        out = torch.bmm(v, self.dropout(attn).permute(0, 2, 1))
        return out, attn


class MultiHeadAttention2D(nn.Module):
    def __init__(self, in_dim, n_head, d_k, d_v, dist_func="cosine", activation=Mish(),
                 batch_norm=nn.BatchNorm2d, use_gamma=False):
        super().__init__()
        self.n_head = n_head
        self.dist_func = dist_func
        self.d_k = d_k
        self.d_v = d_v
        self.use_gamma = use_gamma
        # self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)

        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # assert in_dim % n_head == 0, "group convolution will throw runtime error"
        nn.Conv2d(in_channels=in_dim, out_channels=n_head * d_k, kernel_size=1)
        self.query_conv = Conv_Block(in_dim, filters=n_head * d_k, kernel_sizes=1, stride=1, padding=0,
                                     groups=n_head, activation=activation, batch_norm=batch_norm)
        self.key_conv = Conv_Block(in_dim, filters=n_head * d_k, kernel_sizes=1, stride=1, padding=0,
                                   groups=n_head, activation=activation, batch_norm=batch_norm)
        self.value_conv = Conv_Block(in_dim, filters=n_head * d_v, kernel_sizes=1, stride=1, padding=0,
                                     groups=n_head, activation=activation, batch_norm=batch_norm)
        self.attention = ScaledDotProductAttention(temperature=(in_dim / n_head) ** 0.5)
        self.final_conv = Conv_Block(n_head * d_v, filters=in_dim, kernel_sizes=1, stride=1, padding=0,
                                     groups=1, activation=activation, batch_norm=batch_norm)
        if use_gamma:
            self.gamma = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                B : is the multiplication of batch size and seq length
                C : is the channel size of extracted feature
                W & H : the size of extracted feature
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        b, c, h, w = x.size()
        residual = x
        # split the grouped convolution
        query = self.query_conv(x).view(b, self.n_head, self.d_k, h * w)  # .permute(0, 1, 3, 4, 2)
        key = self.key_conv(x).view(b, self.n_head, self.d_k, h * w)  # .permute(0, 1, 3, 4, 2)
        value = self.value_conv(x).view(b, self.n_head, self.d_v, h * w)  # .permute(0, 1, 3, 4, 2)

        output, attn_map = [], []
        # The reason to use a for loop is to save memory
        for i in range(self.n_head):
            q, k, v = query[:, i, :, :], key[:, i, :, :], value[:, i, :, :]
            out, attn = self.attention(q, k, v, self.dist_func)
            output.append(out)
            attn_map.append(attn)
        output = torch.cat(output, dim=1)
        output = output.view(b, self.n_head * self.d_v, h, w)
        attn_map = extract_attn_map(attn_map).view(b, h, w)
        if self.use_gamma:
            output = self.gamma * self.final_conv(output) + residual
        else:
            output = self.final_conv(output) + residual
        return output, attn_map


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, in_dim, n_head, d_k, d_v, dropout=0.1, dist_func="matmul", use_gamma=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dist_func = dist_func

        self.w_qs = nn.Linear(in_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(in_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(in_dim, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, in_dim, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        if use_gamma:
            self.gamma = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.layer_norm(x)
        residual = x
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q = x.size(0), x.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x n x lq x dv
        query = self.w_qs(x).view(sz_b, len_q, n_head, d_k).transpose(1, 3)
        key = self.w_ks(x).view(sz_b, len_q, n_head, d_k).transpose(1, 3)
        value = self.w_vs(x).view(sz_b, len_q, n_head, d_v).transpose(1, 3)

        output, attn_map = [], []
        for i in range(n_head):
            q, k, v = query[:, :, i, :], key[:, :, i, :], value[:, :, i, :]
            out, attn = self.attention(q, k, v, self.dist_func)
            output.append(out)
            attn_map.append(attn)
        output = torch.cat(output, dim=1).transpose(1, 2)
        if self.use_gamma:
            output = self.dropout(self.fc(output)) * self.gamma
        else:
            output = self.dropout(self.fc(output))
        output += residual
        attn_map = extract_attn_map(attn_map)
        return output, attn_map


def extract_attn_map(attn_map):
    with torch.no_grad():
        attn_map = torch.mean(torch.stack(attn_map, dim=1), dim=1)
        attn_map = torch.sum(attn_map, dim=1)
        min_along_batch = torch.min(attn_map, dim=1)[0].unsqueeze(1).repeat(1, attn_map.size(1))
        max_along_batch = torch.max(attn_map, dim=1)[0].unsqueeze(1).repeat(1, attn_map.size(1))
        attn_map = (attn_map - min_along_batch) / (max_along_batch - min_along_batch)
        return attn_map


if __name__ == '__main__':
    self_attn = ScaledDotProductAttention(1)
    x = (7,)
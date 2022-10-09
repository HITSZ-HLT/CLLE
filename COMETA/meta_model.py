# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss



class Loss_predict_net(nn.Module):
    def __init__(self,d_model=128):
        self.dropout = 0.5
        self.num_classes = 1
        self.n_vocab = 128112
        self.embed = d_model
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256

        super(Loss_predict_net, self, ).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        self.act_fn = nn.Softplus()
        self.ln = torch.nn.LayerNorm(self.embed, elementwise_affine=True)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, emb):
        # [bs,128k, d] ->[bs,1,128k, d]
        out = emb.unsqueeze(1)
        out = self.ln(out)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.act_fn(out)
        return out

    def num_parameters(self,):
        total_num = sum(p.numel() for p in self.parameters())
        # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_num






class Loss_predict_net_dynamic(nn.Module):
    def __init__(self,d_model=128):
        self.dropout = 0.5
        self.num_classes = 1
        self.n_vocab = 128112
        self.embed = d_model
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256

        super(Loss_predict_net_dynamic, self, ).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        self.act_fn = nn.Softplus()
        self.ln = torch.nn.LayerNorm(self.embed, elementwise_affine=True)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, emb):
        bs,n,d=emb.shape
        assert d % self.embed == 0
        emb = emb.view([bs,n,-1,self.embed])
        emb = emb.mean(dim=-2)

        # [bs,128k, d] ->[bs,1,128k, d]
        out = emb.unsqueeze(1)
        out = self.ln(out)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.act_fn(out)
        return out

    def num_parameters(self,):
        total_num = sum(p.numel() for p in self.parameters())
        # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_num



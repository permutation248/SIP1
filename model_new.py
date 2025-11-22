import torch
import torch.nn as nn 
import random
import copy
import math
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
# 核心组件的导入（假设 MLP, VAE, Qc_inference_mlp 已定义或可导入）
from model_VAE_new import VAE


def gaussian_reparameterization_var(means, var, times=1):
    """用于从高斯分布 N(means, var) 中进行重参数化采样 (Reparameterization Sampling)。"""
    std = torch.sqrt(var+1e-8)
    assert torch.sum(std<0).item()==0
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times

def Init_random_seed(seed=0):
    """初始化随机种子以确保实验可复现性。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    """多层感知机 (Multi-Layer Perceptron) 模块，作为编码器的特征提取骨干。"""
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps =[]
        layers = []
        
        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
        return x

class Qc_inference_mlp(nn.Module):
    """标签原型随机编码器：将输入（标签初始嵌入）映射到高斯分布的均值和方差。"""
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(Qc_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim) # 均值 (mu) 头部
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus()) # 方差 (sca) 头部，使用 Softplus 确保非负

    def forward(self, x):
        assert torch.sum(torch.isnan(x)).item() == 0
        hidden_features = self.mlp(x)
        c_mu = self.z_loc(hidden_features) # 均值 (mu)
        c_sca = self.z_sca(hidden_features) # 方差 (sca)
        if torch.sum(torch.isnan(c_mu)).item() >0:
            pass
        assert torch.sum(torch.isinf(c_mu)).item() == 0
        return c_mu, c_sca

class Net(nn.Module):
    """SIP (Semantic Invariance learning and Prototype modeling) 模型主体。"""
    def __init__(self, d_list,num_classes,z_dim,adj,rand_seed=0):
        super(Net, self).__init__()
        self.rand_seed = rand_seed
        
        # Label semantic encoding module
        # 标签原型的初始基底，可学习的单位矩阵（C x C），作为原型 VAE 的输入
        self.label_embedding_u = nn.Parameter(torch.eye(num_classes),
                                             requires_grad=True)
        # 标签原型的初始标准差（未在最终前向传播中使用）
        self.label_embedding_std = nn.Parameter(torch.ones(num_classes),
                                             requires_grad=True)
        # 标签图邻接矩阵的初始值，可学习（未在最终前向传播中使用）
        self.label_adj = nn.Parameter(torch.eye(num_classes),
                                      requires_grad=True)
        self.adj = adj
        self.z_dim = z_dim
        self.label_mlp = Qc_inference_mlp(num_classes, z_dim) # 标签原型随机编码器
        
        # GMM 先验参数（未在最终损失中使用，但定义在模型结构中）
        self.mix_prior = None
        self.mix_mu = None
        self.mix_sca = None
        self.k = num_classes
        
        # VAE module - 信息瓶颈框架的核心
        self.VAE = VAE(d_list=d_list,z_dim=z_dim,class_num=num_classes)
        
        # Classifier - 最终分类匹配器 (分组 1D 卷积)
        self.cls_conv = nn.Conv1d(num_classes, num_classes,
                                  z_dim*2, groups=num_classes)
        
        # 调用 set_prior 初始化 GMM 参数并转移到 GPU
        self.set_prior()
        self.cuda()
    
    def set_prior(self):
        """初始化 GMM 先验分布的参数（均值、方差和权重）。"""
        self.mix_prior = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        self.mix_mu = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
        self.mix_sca = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)

    def forward(self, x_list,mask):
        # Generating semantic label embeddings via label semantic encoding module
        label_embedding = self.label_embedding_u
        
        # 标签原型随机编码器推断均值和方差
        label_embedding, label_embedding_var = self.label_mlp(self.label_embedding_u)
        
        # 对标签原型分布进行重参数化采样 (times=10)
        label_embedding_sample = gaussian_reparameterization_var(label_embedding,label_embedding_var,10)
        
        # VAE 核心：融合多视图特征并生成共享表示 z_sample
        # xr_list是各个视图的重构结果,x_list是各个视图的输入数据
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list = self.VAE(x_list,mask)
        
        
        # ###LxZ5: 最终分类预测
        # 1. 特征拼接: 将样本的共享特征 z_sample 与所有标签原型 label_embedding_sample 拼接
        # 维度: (N, C, D_z) + (N, C, D_z) -> (N, C, 2*D_z)
        qc_z = torch.cat((z_sample.unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),
                          label_embedding_sample.unsqueeze(0).repeat(z_sample.shape[0],1,1)),dim=-1)

        # 2. 分组卷积匹配: 通过 cls_conv (1D Conv) 得到分类得分
        # 输入 (N, C, 2*D_z)，输出 (N, C, 1)，然后移除长度为 1 的维度
        p = self.cls_conv(qc_z).squeeze(-1)

        # 3. Sigmoid 激活: 得到最终预测概率
        p = torch.sigmoid(p)
        
        # 返回所有关键变量 (用于计算 L_CE, L_rec, L_PA, L_sha)
        return z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_embedding_sample,p, label_embedding, label_embedding_var, None

def get_model(d_list,num_classes,z_dim,adj,rand_seed=0):
    """模型初始化工厂函数。"""
    model = Net(d_list,num_classes=num_classes,z_dim=z_dim,adj=adj,rand_seed=rand_seed)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() 
                                    else 'cpu'))
    return model
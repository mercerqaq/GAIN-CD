import torch
import torch.nn as nn
import os,pickle
import numpy as np
import pandas as pd
#import ot
import torch.nn.functional as F
#from utils.arch import MLP, AttentionMLP
from utils.arch import MLP
from utils.missing import nanmean
from dagma import DagmaMLP
from tqdm.auto import tqdm
from imputers import OTimputer
from GAINImputer import GAINImputer
from  sklearn.metrics import  mean_squared_error
#from main_letter_spam import gain
import typing, copy
import logging
from utils.eval import evaluate, write_result
from config import get_data

def exact_ot_cost(X, Y, cost_fn = 'euclidean'):#计算OT成本，cost_fn计算两个集合中两个点之间的距离，这里使用欧几里得距离
    batchsize, _  = X.shape
    Y = Y.to(X.device)
    unif = torch.ones((batchsize,), device = X.device) / batchsize#表示每个数据点在最优传输中占的权重都是均匀的
    if cost_fn == 'euclidean':#如果传入参数是欧几里得距离，那么计算X和Y之间的欧几里得距离
        M = torch.cdist(X, Y, p=2)
        M = torch.sqrt(M)
    else:#############如果不使用欧几里得距离，这里允许使用其他计算X和Y之间距离的自定义方法，M是创建的一个零矩阵用来存放距离，cost_fn是自定义的计算距离方法，##这里没有定义，需要自己写
        M = torch.zeros((batchsize, batchsize), device = X.device)
        for i in range(batchsize): 
            for j in range(batchsize):
                M[i,j] = cost_fn(X[i:i+1, ], Y[j:j+1, ])
    loss = ot.emd2(unif, unif, M)#用ot.emd2方法计算两组数据点之间的wasserstein距离，计算OT损失的这个方法在POT库已经写好了
    return loss #返回OT损失


def rbf_kernel(X, Y):#基于RBF核衡量数据点之间的相似度
    batch_size, h_dim = X.shape
    X = X.to(Y.device)
    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x
    
    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y
    
    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd
    
    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:#控制RBF核的扩展程度
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x) + torch.exp(-C * dists_y)

        res1 = (1 - torch.eye(batch_size).to(X.device)) * res1
        
        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats / batch_size#这个函数最终基于RBF核计算了一个相似度统计量可能用于正则化或者损失函数等


class SuperImputer(nn.Module): #插补缺失数据的类，主要是基于MLP预测填补值
    def __init__(self, data, mask, hidden_dims, sem_type, initialized = None):#初始化参数，hidden_dims存放使用的神经网络隐藏层维度，initialized表示初始化缺失值的方法，这里取None表示用0填充，如果用learnable则表示用学习到的可训练值填充缺失
        super(SuperImputer, self).__init__()#继承父类nn.Module，nn.Module是Pytorch库中定义的一个类
        print('Using Super imputation ...')
        self.D = hidden_dims[0]        
        self.mu = MLP(hidden_dims, nn.ReLU())#用两个MLP网络按gaosifenbucaiyang
        self.var = MLP(hidden_dims, nn.ReLU())

        self.data = data.to(torch.float32)  # 确保数据是 float32 类型
        self.mask = mask.to(torch.float32)
        self.initialized = initialized
        self.sem_type = sem_type
 
        if initialized == 'learnable':
            imps = (torch.randn(data.shape, device = mask.device) + nanmean(data, 0))[mask.bool()]
            self.imps = nn.Parameter(imps)
    
    def forward(self): #前向传播
        
        
        x = self.data.clone()
        if self.initialized is None: # 选择初始化缺失值的方式：零填充或者用训练得到的值填充
            x[self.mask.bool()] = 0.0

        elif self.initialized == 'learnable':
            x[self.mask.bool()] = self.imps

        #mu_out = self.mu(x)  # ← 均值预测
        logvar = self.var(x)#使用mlp计算数据方差，返回了方差的对数

        if self.sem_type == 'sachs' or 'dream' in self.sem_type:#根据sem的类型生成填补值
            imps = self.mu(x) + torch.square(0.5 * logvar) * torch.randn_like(x)
        else:####torch.randn_like(x)的作用是生成一个和数据x形状相同的正态分布的噪声，torch.square(0.5 * logvar)是把方差对数的一半平方，torch.square(0.5 * logvar) * torch.randn_like(x)意味着生成了一个由方差控制的噪声，mu(x)是用mlp计算的x的均值，全部加在一起即用均值加上一个噪声来填补缺失。
            imps = self.mu(x) + torch.exp(0.5 * logvar) * torch.randn_like(x)#和上边的不同就是在sem不同时，这里用标准差计算。
        
        if self.sem_type in ('sachs', 'neuro'):#如果 sem_type 是 'sachs' 或 'neuro'，对插补值 imps 应用 ReLU 和 tanh 激活函数。这可能是为了确保插补值符合特定的约束，如非负性或有特定的分布。
            imps = torch.relu(torch.tanh(imps))
            
        x = imps * self.mask + x#填补缺失  这里说明对mask要用1表示缺失，0表示不缺失
        #return x,mu_out #返回填补后的数据
        return x

class OTSuperImputer(OTimputer):###调用OT填补方法 ######################################使用OT插补的SuperImputer############################1.2日添加的
    def __init__(self, data, mask, hidden_dims, sem_type, initialized=None, device='cuda', **kwargs):
        super(OTSuperImputer, self).__init__(**kwargs)
        self.device = device  
        self.data = data.to(device)  
        self.mask = mask.to(device)  
        self.sem_type = sem_type

    def forward(self):
        # 调用 OTimputer 的 fit_transform 方法
        X_filled = self.fit_transform(self.data.clone())
        return X_filled



'''
class GAINSuperImputer(GAINImputer):
    def __init__(self, data, mask, hidden_dims, sem_type, alpha=10, iterations=10000, batch_size=128, hint_rate=0.9, device='cuda'):
        super(GAINSuperImputer, self).__init__(
            data=data, mask=mask, hidden_dims=hidden_dims, alpha=alpha,
            iterations=iterations, batch_size=batch_size, hint_rate=hint_rate, device=device
        )
        self.sem_type = sem_type

    def forward(self, x=None, m=None):
        # 调用 GAIN 模型的 forward
        hat_x, d_prob = super().forward(x, m)

        print(f"Filled data (hat_x): {hat_x}")
        print(f"Discriminator probability (d_prob): {d_prob}")

        return hat_x, d_prob
'''
'''
class GAINSuperImputer():#####需要改成能反馈的GAIN
    def __init__(self, data, mask, hidden_dims, sem_type, alpha=100, iterations=10000, batch_size=128, hint_rate=0.9,
                 device='cuda'):
        self.data = data
        self.mask = mask
        self.hidden_dims = hidden_dims
        self.sem_type = sem_type
        self.alpha = alpha
        self.iterations = iterations
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.device = device

    def forward(self):
        # 调用 main_letter_spam.py 中的 gain 函数进行填补
        # gain 函数需要 miss_data_x 和 gain_parameters
        gain_parameters = {'batch_size': self.batch_size, 'hint_rate': self.hint_rate, 'alpha': self.alpha,
                           'iterations': self.iterations}

        miss_data_x_df = pd.read_csv('dataset9-miss.csv', header=None)  # 读取缺失数据
        miss_data_x = miss_data_x_df.to_numpy()

        # 这里我们传递 miss_data_x 和 gain_parameters 来调用 gain 函数
       # imputed_data_x = gain(miss_data_x, gain_parameters,model_save_path='D:\pycharmproject\OTM-CPU\model_path') # 执行填补
        imputed_data_x_df= pd.read_csv('dataset9-impute.csv', header=None)
        imputed_data_x = imputed_data_x_df.to_numpy()
        imputed_data_x_tensor = torch.tensor(imputed_data_x, dtype=torch.float64).to(self.device)

        print(f"Filled data (hat_x): {imputed_data_x}")
        return imputed_data_x_tensor, None  # 返回填补数据(张量)，这里不关心判别器的输出 d_prob
'''
'''
####加了逆概率加权后的mlp的missmodel
class MissModel(nn.Module):
    def __init__(self, data, mask, hidden_dims, device, sem_type, initialized='learnable'):
        super().__init__()
        self.d = hidden_dims[0]
        self.sem_type = sem_type
        self.device = device

        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)

        # MLP 填补器：现在会返回 (x_filled, mu_out)
        if sem_type == 'neuro' or 'dream' in sem_type:
            self.imputer = SuperImputer(data, mask, [self.d, self.d, self.d], sem_type, initialized)
        else:
            self.imputer = SuperImputer(data, mask, [self.d, self.d], sem_type, initialized)

        # 缺失概率网络：输入 d，输出 d，经 sigmoid 得 P(obs=1)
        self.miss_net = MLP([self.d, self.d, self.d])
        self.sigmoid  = nn.Sigmoid()

        # 保存原始
        self.data = data
        self.mask = mask  # 1=缺失, 0=观测

    def to_adj(self):
        return self.scm.fc1_to_adj()

    def forward(self):
        # 1) 插补
        x_filled, mu_out = self.imputer()

        # 2) 结构重建
        xhat = self.scm(x_filled)

        # 3) 观测概率（标签=1-mask）
        obs_mask = 1.0 - self.mask                  # 1=观测
        pi_hat   = self.sigmoid(self.miss_net(x_filled))  # [n,d]

        # 统一返回，训练里用
        return x_filled, xhat, mu_out, pi_hat, obs_mask, self.data
'''


####插补调用

class MissModel(nn.Module):##MLP填补的MissModel
    def __init__(self, data, mask, hidden_dims, device, sem_type, initialized=None):  # 初始的initialized参数选择None
        super(MissModel, self).__init__()  # 继承父类

        self.d = hidden_dims[0]  # 隐藏层维度
        self.sem_type = sem_type
        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)  ####通过DagmaMLP类初始化了DAGMA模型scm，用于生成模型的重构

        if sem_type == 'neuro' or 'dream' in sem_type:  ###根据sem类型调用SuperImputer类进行初始插补
            self.imputer = SuperImputer(data.to(torch.float32), mask.to(torch.float32), [self.d, self.d, self.d], sem_type, initialized)
        else:
            self.imputer = SuperImputer(data.to(torch.float32), mask.to(torch.float32), [self.d, self.d], sem_type, initialized)

    def to_adj(self):
        return self.scm.fc1_to_adj()  # 将模型输出转化为一个邻接矩阵

    def forward(self):
        #global X_gain#9.4xinjia

        #X_gain = _load_X_true_via_get_data("/home/sunbaodan/PycharmProjects/OTM-CPU/dataset/MLP-ER11.pickle")
        #x=X_gain

        x = self.imputer()#9.4xinjian
        # reconstruction from the imputations
        xhat = self.scm(x)
        return x, xhat  # 前向传播，使用SuperImputer生成插补数据x，使用DAGMA对插补后的x重构生成新的输出xhat


'''
class MissModel(nn.Module):##OT填补的MissModel
    def __init__(self, data, mask, hidden_dims, device, sem_type, initialized = None):#初始的initialized参数选择None
        super(MissModel, self).__init__()#继承父类

        self.d = hidden_dims[0]#隐藏层维度
        self.sem_type = sem_type
        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)####通过DagmaMLP类初始化了DAGMA模型scm，用于生成模型的重构
        self.imputer = OTSuperImputer(
            data=data,
            mask=mask,
            hidden_dims=hidden_dims,
            sem_type=sem_type,
            eps=0.01,  # OT 参数
            lr=1e-2,  # OT 学习率
            niter=2000,  # 最大迭代次数
            batchsize=128,
            n_pairs=10,
            device=device,
        )

    def to_adj(self):
        return self.scm.fc1_to_adj()#将模型输出转化为一个邻接矩阵

    def forward(self):


        x = self.imputer.forward()
        # reconstruction from the imputations
        xhat = self.scm(x)
        return x, xhat#前向传播，使用OTSuperImputer生成插补数据x，使用DAGMA对插补后的x重构生成新的输出xhat
'''

'''
##Gain插补的MissModel
class MissModel(nn.Module):
    def __init__(self, data, mask, hidden_dims, device, sem_type, initialized=None):
        super(MissModel, self).__init__()

        self.d = hidden_dims[0]
        self.sem_type = sem_type
        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)  # DAGMA 结构学习部分

        # 使用 GAINSuperImputer 进行数据插补
        self.imputer = GAINSuperImputer(
            data=data,
            mask=mask,
            hidden_dims=hidden_dims,
            sem_type=sem_type,
            alpha=10,  # GAIN 特定参数
            iterations=10000,
            batch_size=128,
            hint_rate=0.9,
            device=device
        )

    def to_adj(self):
        return self.scm.fc1_to_adj()

    def forward(self):
        # 调用插补器，确保返回的是填补后的数据 Tensor
        x, _ = self.imputer.forward()  # 忽略判别器输出 d_prob
        # reconstruction from the imputations
        xhat = self.scm(x)
        return x, xhat
'''

'''
#Gain8月7号的填补missmodel#1216 
class MissModel(nn.Module):
    def __init__(self, data, mask, hidden_dims, device, sem_type, miss_rate=0.1, seed=13):
        super(MissModel, self).__init__()
        self.data = data
        self.d = hidden_dims[0]
        self.device = device
        self.sem_type = sem_type
        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)  # DAGMA 结构学习部分

        # 修复初始化GAINImputer时缺少sem_type的传递
        self.imputer = GAINImputer(data, mask, hidden_dims, sem_type, device=device, miss_rate=miss_rate)

    def to_adj(self):
        return self.scm.fc1_to_adj()

    def forward(self,X_true=None):
        # 调用train_and_impute进行填补
        x_imputed_df=pd.read_csv('0120MIP50-ER2ximputed.csv',header=None)
        x_imputed = x_imputed_df.to_numpy()
        # 通过DAGMA模块进行重建
        #W_est = self.to_adj()
        # 确保填补数据是 PyTorch 张量
        #if isinstance(x_imputed, np.ndarray):
        #    x_imputed = torch.tensor(x_imputed, dtype=torch.float32).to(self.device)  # 转换为张量并移到设备上
        # 如果提供了真实数据 X_true，则计算 MSE
        #if X_true is not None:
            # 确保 X_true 和填补数据都在同一个设备上，并且断开计算图
        #    mse = mean_squared_error(
         #       X_true.cpu().detach().numpy(),  # 真实数据
                #      x_imputed.cpu().detach().numpy()  # 填补数据
            #)
           # print(f'MSE between imputed data and true data: {mse}')
        xhat = self.scm(x_imputed)
        #print(f'GAIN的填补', x_imputed)
        #print(f'GAIN的重建', xhat)
        return x_imputed, xhat
'''
'''
#Gain0115想法：把GAIN的训练放在总体联合框架下，调用填补的时候不训练只用现有生成器生成填补数据.GAINImputer.py add impute_once code
class MissModel(nn.Module):
    def __init__(self, data, mask, hidden_dims, device, sem_type, miss_rate=0.1, seed=13):
        super(MissModel, self).__init__()
        self.data = data
        self.d = hidden_dims[0]
        self.device = device
        self.sem_type = sem_type
        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)  # DAGMA 结构学习部分

        # 修复初始化GAINImputer时缺少sem_type的传递
        self.imputer = GAINImputer(data, mask, hidden_dims, sem_type, device=device, miss_rate=miss_rate)

    def to_adj(self):
        return self.scm.fc1_to_adj()

    def forward(self,X_true=None):
        # 调用train_and_impute进行填补
        x_imputed = self.imputer.train_and_impute(self.data)  # 调用GAINImputer进行训练和填补
        ##x_imputed_df=pd.read_csv('1218MIMximputed.csv',header=None)
        ##x_imputed = x_imputed_df.to_numpy()
        # 通过DAGMA模块进行重建
        #W_est = self.to_adj()
        # 确保填补数据是 PyTorch 张量
        #if isinstance(x_imputed, np.ndarray):
        #    x_imputed = torch.tensor(x_imputed, dtype=torch.float32).to(self.device)  # 转换为张量并移到设备上
        # 如果提供了真实数据 X_true，则计算 MSE
        #if X_true is not None:
            # 确保 X_true 和填补数据都在同一个设备上，并且断开计算图
        #    mse = mean_squared_error(
         #       X_true.cpu().detach().numpy(),  # 真实数据
                #      x_imputed.cpu().detach().numpy()  # 填补数据
            #)
           # print(f'MSE between imputed data and true data: {mse}')


        xhat = self.scm(x_imputed)
        #print(f'GAIN的填补', x_imputed)
        #print(f'GAIN的重建', xhat)
        return x_imputed, xhat
'''
'''
class MissModel(nn.Module):##dancitianbudeGAIN
    def __init__(self, data, mask, hidden_dims, device, sem_type,miss_rate=0.1, seed=13, reuse_gain=True, gain_once=True):
        super(MissModel, self).__init__()
        self.data = data
        self.d = hidden_dims[0]
        self.device = device
        self.sem_type = sem_type
        self.reuse_gain = reuse_gain     # True: 缓存后复用
        self.gain_once = gain_once       # True: 只跑一次 GAIN
        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)

        # GAIN 插补器（仅用于生成 X_gain，不参与反传）
        self.imputer = GAINImputer(data, mask, hidden_dims, sem_type,device=device, miss_rate=miss_rate)
        # 用于缓存 GAIN 的填补（注册成 buffer，自动随 device 移动）
        self.register_buffer("X_gain", None)
    def to_adj(self):
        return self.scm.fc1_to_adj()

    def _ensure_gain_cached(self):
        """如有需要，训练一次 GAIN 并缓存结果到 self.X_gain（不可微）"""
        if (self.X_gain is None) or (not self.reuse_gain):
                x_np_or_t = self.imputer.train_and_impute(self.data)  # 可能返回 numpy 或 tensor
                if isinstance(x_np_or_t, np.ndarray):
                    x_t = torch.from_numpy(x_np_or_t).float()
                else:
                    x_t = x_np_or_t.float()
                # 形状/设备/梯度安全
                assert x_t.dim() == 2 and x_t.size(1) == self.d, \
                    f"X_gain shape {tuple(x_t.shape)} != (_, {self.d})"
                x_t = x_t.to(self.device).detach()
                x_t.requires_grad_(False)
                self.X_gain = x_t  # 缓存
            # 如果只允许跑一次，之后就不再调用 GAIN
                if self.gain_once:
                    if hasattr(self.imputer, "parameters"):
                        for p in self.imputer.parameters():
                            p.requires_grad = False

    def forward(self):
        # 1) 确保 X_gain 已缓存
        self._ensure_gain_cached()
        x = self.X_gain
        # 3) DAGMA 重建
        xhat = self.scm(x)

        # 调试时不要打印整个矩阵，容易卡 stdout；可打印形状/均值
        # print("GAIN X:", x.shape, x.mean().item(), "Xhat:", xhat.shape, xhat.mean().item())
        return x, xhat
'''

logging.basicConfig(filename='training_loss.log',  # 指定日志文件路径
                    level=logging.INFO,           # 设置日志级别为INFO
                    format='%(asctime)s - %(message)s')  # 设置日志格式（包括时间戳）
class DagmaNonlinear:#非线性DAGMA
    """
    Class that implements the DAGMA algorithm
    """
    
    def __init__(self, model: nn.Module,X_true=None, device=None,verbose: bool = False, dtype: torch.dtype = torch.float32):
        
        self.vprint = print if verbose else lambda *a, **k: None#verbose控制是否打印调试信息
        self.device = device
        self.model = model
        self.X_true = X_true
        self.dtype = dtype
        self.cached_X = None
        self.cached_Xhat = None
        self.fixed_X=None#9.4xinjia


        self.data = self.model.imputer.data#从插补器中提取数据和掩码矩阵
        self.mask = self.model.imputer.mask
    
    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = output.device  # 获取 output 的设备
        target = target.to(device)

        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))#计算输出和目标之间的对数均方误差损失
        return loss

    def minimize(self,                   ########优化步骤的实现，通过最小化一个目标函数来优化模型参数
                 max_iter: float, #最大迭代次数
                 lr: float, #学习率
                 lambda1: float, #正则化参数
                 lambda2: float, #正则化参数
                 mu: float, 
                 s: float,
                 lr_decay: float = False, #是否启用学习率衰减
                 tol: float = 1e-6, #容忍度
                 pbar: typing.Optional[tqdm] = None,#进度条
        ) -> bool:
        self.vprint(f'\nMinimize s={s} -- lr={lr}')#打印优化信息
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)###使用Adam优化器优化模型参数，由前边定义的一些参数控制

        if lr_decay is True:#启用指数学习率衰减
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16#初始化目标函数的前一个值为一个非常大的数。
        for i in range(max_iter):########开始迭代##########
            
            h_val = self.model.scm.h_func(s)#调用模型的 scm.h_func(s) 计算某个与 s 相关的值 h_val，这个值可能是模型中的约束条件。如果 h_val 小于零，则输出警告并返回 False，停止优化。
            print(f"h_val  device: {h_val.device}, dtype: {h_val.dtype}")
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            print('内层minimize循环')
            #if self.cached_X is None:
            #   X, Xhat = self.model()  # 这里会跑 GAIN
            #    self.cached_X = X
            #    self.cached_Xhat = Xhat
            #else:
            #    X, Xhat = self.cached_X, self.cached_Xhat

            X, Xhat = self.model()#获取模型的输入X和重建输出Xhat
            #X=torch.from_numpy(X)##GAIN填补的时候取消注释
            X = X.to(self.device).float()  # 确保X为float32类型
            Xhat = Xhat.to(self.device).float()
            score = self.log_mse_loss(Xhat, X)##计算MSE
            l1_reg = lambda1 * self.model.scm.fc1_l1_reg().to(self.device).float()

            if self.model.sem_type == 'mlp':
                obj = mu * (score + l1_reg) + h_val + 0.01 * rbf_kernel(Xhat, X)#0.001*exact_ot_cost(Xhat, X,cost_fn = 'euclidean')#+######目标函数的组成包括重建误差score，L1正则化项，h_val约束，RBF核正则化项rbf_kernel(Xhat, X)，模型的一个超参数mu
            else:
                obj = mu * (score + l1_reg) + h_val +1.5 *  rbf_kernel(Xhat, X)###目标函数计算
                obj = obj.to(self.device).float()

            #if i % 500 == 0:ianzhushidiaoshuchuzhongjiansunshiderizhi
            #    # 打印到控制台
            #    print(f"Iteration {i}, Loss: {obj.item()}")
            #    WW_est = self.model.to_adj()
            #    WW_est = torch.tensor(WW_est)
            #    WW_est = WW_est.cpu().detach().numpy()  #######运行到这一步的时候出错了，提示W_est在这一行之前是numpy数组而不是pytorch张量，不能用.cpu()方法，解决办法是在前一行加上W_est = torch.tensor(W_est)把它转换成张量，还没试可不可以
            #    WW_est[np.abs(WW_est) < 0.3] = 0  #
            #    # 将损失写入日志文件
            #    logging.info(f"Iteration {i}, Loss: {obj.item()}")
            #    zhongjian_result = evaluate(dataset.B_bin, WW_est, threshold=0.3)


            #    saved_path = f'output/otmWEST.txt'
            #    write_result(zhongjian_result, code,saved_path)
            #####反向传播和优化
            optimizer.zero_grad()#清除梯度避免累积
            obj.backward(retain_graph=True)#计算目标函数的梯度
            optimizer.step()#更新模型参数，包括模型权重、偏置、可训练参数等
            
            if lr_decay and (i+1) % 1000 == 0: #如果启用了学习率衰减，每 1000 次迭代更新一次学习率
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:#检查收敛
                obj_new = obj.item()#获取当前目标函数值
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:#如果目标函数变化小于给定的容忍度 tol，则认为收敛，跳出循环；否则，继续优化并更新进度条。
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return True#优化结束

    def fit(self, 
            lambda1: float = .02, 
            lambda2: float = .005,
            T: int = 4, 
            mu_init: float = .1, 
            mu_factor: float = .1, 
            s: float = 1.0,
            warm_iter: int = 5e4, 
            max_iter: int = 8e4, 
            lr: float = .0002, 
            w_threshold: float = 0.3, 
            checkpoint: int = 1000,
        ) -> np.ndarray:

    ########用于训练DAGMA模型，最终生成估计的邻接矩阵 W_est
    
        
        
        self.checkpoint = checkpoint
        mu = mu_init#权重参数
        if type(s) == list:#处理s参数，如果 s 是一个列表且其长度小于 T（迭代次数），则使用列表的最后一个值来填充剩余部分；如果 s 是整数或浮动值，则将 s 复制 T 次，保证每个迭代中都有一个对应的值。
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:####训练过程x
            for i in range(int(T)):#训练的阶段数，4个阶段每次训练使用不同的超参数。
                print(f'外层T循环')
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]#：设置 success 为 False，并获取当前 s 值（每次迭代不同）。
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)#每个训练周期的迭代次数
                model_copy = copy.deepcopy(self.model)
                lr_decay = False#选择是否启用学习率衰减
                while success is False:#self.minimize 方法执行优化步骤，如果优化失败（success is False），恢复模型的状态，并减小学习率（lr *= 0.5）。如果学习率降到 1e-10 以下，则停止训练。
                    success = self.minimize(inner_iter, lr, lambda1, lambda2, mu, s_cur,
                                        lr_decay, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy()) 
                        lr *= 0.5 
                        lr_decay = True
                        if lr < 1e-10:
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor
        
        W_est = self.model.to_adj()#将模型的输出转换为邻接矩阵
        W_est = torch.tensor(W_est)#将得到的邻接矩阵转换为 PyTorch 张量
        W_est = W_est.cpu().detach().numpy()#######运行到这一步的时候出错了，提示W_est在这一行之前是numpy数组而不是pytorch张量，不能用.cpu()方法，解决办法是在前一行加上W_est = torch.tensor(W_est)把它转换成张量，还没试可不可以
        W_est[np.abs(W_est) < w_threshold] = 0##设置一个阈值，将邻接矩阵中绝对值小于 w_threshold 的元素置为零，去除弱连接。
        
        return W_est#得到邻接矩阵

'''
logging.basicConfig(filename='training_loss.log',  # 指定日志文件路径
                    level=logging.INFO,           # 设置日志级别为INFO
                    format='%(asctime)s - %(message)s')  # 设置日志格式（包括时间戳）
class DagmaNonlinear:#非线性DAGMA
    """
    Class that implements the DAGMA algorithm
    """

    def __init__(self, model: nn.Module,X_true=None, device=None,verbose: bool = False, dtype: torch.dtype = torch.float32):

        self.vprint = print if verbose else lambda *a, **k: None#verbose控制是否打印调试信息
        self.device = device
        self.model = model
        self.X_true = X_true
        self.dtype = dtype


        self.data = self.model.imputer.data#从插补器中提取数据和掩码矩阵
        self.mask = self.model.imputer.mask

    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = output.device  # 获取 output 的设备
        target = target.to(device)

        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))#计算输出和目标之间的对数均方误差损失
        return loss

    def minimize(self,                   ########优化步骤的实现，通过最小化一个目标函数来优化模型参数
                 max_iter: float, #最大迭代次数
                 lr: float, #学习率
                 lambda1: float, #正则化参数
                 lambda2: float, #正则化参数
                 mu: float,
                 s: float,
                 lr_decay: float = False, #是否启用学习率衰减
                 tol: float = 1e-6, #容忍度
                 pbar: typing.Optional[tqdm] = None,#进度条
                 beta2: float = 1.0,
                 gamma_bce: float = 0.5
        ) -> bool:
        self.vprint(f'\nMinimize s={s} -- lr={lr}')#打印优化信息

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)###使用Adam优化器优化模型参数，由前边定义的一些参数控制

        if lr_decay is True:#启用指数学习率衰减
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16#初始化目标函数的前一个值为一个非常大的数。
        for i in range(max_iter):########开始迭代##########

            h_val = self.model.scm.h_func(s)#调用模型的 scm.h_func(s) 计算某个与 s 相关的值 h_val，这个值可能是模型中的约束条件。如果 h_val 小于零，则输出警告并返回 False，停止优化。
            print(f"h_val  device: {h_val.device}, dtype: {h_val.dtype}")
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            print('内层minimize循环')
            X, Xhat, MU, PI, OBS, XOBS = self.model()#获取模型的输入X和重建输出Xhat
            #X=torch.from_numpy(X)##GAIN填补的时候取消注释
            #X = X.to(self.device).float()  # 确保X为float32类型
            #Xhat = Xhat.to(self.device).float()
            score = self.log_mse_loss(Xhat, X)##计算MSE
            l1_reg = lambda1 * self.model.scm.fc1_l1_reg().to(self.device).float()


            eps = 1e-6
            # 仅在“该列存在缺失”的列上计算
            cols_with_miss = (self.model.mask.sum(dim=0) > 0)  # [d] bool

            PIc = PI.clamp(1e-6, 1 - 1e-6)
            w = 1.0 / (PIc + eps)  # [n,d]
            num = ((OBS * XOBS) * w).sum(dim=0)  # [d]
            den = (OBS * w).sum(dim=0) + eps  # [d]
            tau_s = num / den  # [d]
            tau_m = MU.mean(dim=0)  # [d]
            R2 = ((tau_s - tau_m) ** 2)[cols_with_miss].sum()
            # === BCE: 监督 miss_net（观测指示）===
            print('PI stats: min', PI.min().item(), 'max', PI.max().item(), 'has_nan', torch.isnan(PI).any().item())
            print('OBS stats: min', OBS.min().item(), 'max', OBS.max().item(), 'dtype', OBS.dtype)
            bce = F.binary_cross_entropy(PIc, OBS)

            # loss=exact_ot_cost(Xhat, X)
            if self.model.sem_type == 'mlp':####计算目标函数，如果sem类型是mlp则有所不同
                obj = mu * (score + l1_reg) + h_val + beta2 * R2 + gamma_bce * bce#+0.001*loss######目标函数的组成包括重建误差score，L1正则化项，h_val约束，RBF核正则化项rbf_kernel(Xhat, X)，模型的一个超参数mu
            else:
                obj = mu * (loss + l1_reg) + h_val +1.5 *  rbf_kernel(X, Xhat)#+0.001*loss##目标函数计算
                obj = obj.to(self.device).float()

            if i % 500 == 0:
                # 打印到控制台
                print(f"Iteration {i}, Loss: {obj.item()}")
                WW_est = self.model.to_adj()
                WW_est = torch.tensor(WW_est)
                WW_est = WW_est.cpu().detach().numpy()  #######运行到这一步的时候出错了，提示W_est在这一行之前是numpy数组而不是pytorch张量，不能用.cpu()方法，解决办法是在前一行加上W_est = torch.tensor(W_est)把它转换成张量，还没试可不可以
                WW_est[np.abs(WW_est) < 0.3] = 0  #
                # 将损失写入日志文件
                logging.info(f"Iteration {i}, Loss: {obj.item()}")
                zhongjian_result = evaluate(dataset.B_bin, WW_est, threshold=0.3)
                saved_path = f'output/otmWEST.txt'
                write_result(zhongjian_result, code,saved_path)
            ####反向传播和优化
            optimizer.zero_grad()#清除梯度避免累积
            obj.backward(retain_graph=True)#计算目标函数的梯度
            optimizer.step()#更新模型参数，包括模型权重、偏置、可训练参数等

            if lr_decay and (i+1) % 1000 == 0: #如果启用了学习率衰减，每 1000 次迭代更新一次学习率
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:#检查收敛
                obj_new = obj.item()#获取当前目标函数值
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:#如果目标函数变化小于给定的容忍度 tol，则认为收敛，跳出循环；否则，继续优化并更新进度条。
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return True#优化结束

    def fit(self,
            lambda1: float = .02,
            lambda2: float = .005,
            T: int = 4,
            mu_init: float = .1,
            mu_factor: float = .1,
            s: float = 1.0,
            warm_iter: int = 5e4,
            max_iter: int = 8e4,
            lr: float = .0002,
            w_threshold: float = 0.3,
            checkpoint: int = 1000,
        ) -> np.ndarray:

    ########用于训练DAGMA模型，最终生成估计的邻接矩阵 W_est



        self.checkpoint = checkpoint
        mu = mu_init#权重参数
        if type(s) == list:#处理s参数，如果 s 是一个列表且其长度小于 T（迭代次数），则使用列表的最后一个值来填充剩余部分；如果 s 是整数或浮动值，则将 s 复制 T 次，保证每个迭代中都有一个对应的值。
            if len(s) < T:
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:####训练过程x
            for i in range(int(T)):#训练的阶段数，4个阶段每次训练使用不同的超参数。
                print(f'外层T循环')
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]#：设置 success 为 False，并获取当前 s 值（每次迭代不同）。
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)#每个训练周期的迭代次数
                model_copy = copy.deepcopy(self.model)
                lr_decay = False#选择是否启用学习率衰减
                while success is False:#self.minimize 方法执行优化步骤，如果优化失败（success is False），恢复模型的状态，并减小学习率（lr *= 0.5）。如果学习率降到 1e-10 以下，则停止训练。
                    success = self.minimize(inner_iter, lr, lambda1, lambda2, mu, s_cur,
                                        lr_decay, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5
                        lr_decay = True
                        if lr < 1e-10:
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor

        W_est = self.model.to_adj()#将模型的输出转换为邻接矩阵
        W_est = torch.tensor(W_est)#将得到的邻接矩阵转换为 PyTorch 张量
        W_est = W_est.cpu().detach().numpy()#######运行到这一步的时候出错了，提示W_est在这一行之前是numpy数组而不是pytorch张量，不能用.cpu()方法，解决办法是在前一行加上W_est = torch.tensor(W_est)把它转换成张量，还没试可不可以
        W_est[np.abs(W_est) < w_threshold] = 0##设置一个阈值，将邻接矩阵中绝对值小于 w_threshold 的元素置为零，去除弱连接。

        return W_est#得到邻接矩阵
'''
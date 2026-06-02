
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from gain import GAIN
from utilsgain import get_mask, EarlyStopper

class GAINImputer:
    def __init__(self, data, mask, hidden_dims, sem_type, device='cuda', miss_rate=0.1):
        self.data = data  # 添加 data 属性
        self.mask = mask
        self.hidden_dims = hidden_dims
        self.sem_type = sem_type
        self.device = device
        self.miss_rate = miss_rate


    def add_missings(self, x: np.array, miss_rate: float) -> np.array:
        """
        Randomly adds NaN values to the data matrix
        """
        x_missed = np.copy(x).astype(float)
        n, m = x.shape
        for j in range(m):
            nan_indexes = np.random.choice(n, int(n * miss_rate), replace=False)
            x_missed[nan_indexes, j] = np.nan
        return x_missed

    def train_and_impute(self, data):
        """
        训练GAIN模型并填补缺失数据
        """
        # 如果数据在GPU上，将其移至CPU并转换为NumPy数组
        data_cpu = data.cpu().numpy() if data.is_cuda else data.numpy()

        # 归一化数据
        scaler = MinMaxScaler()
        X_true_std = scaler.fit_transform(data_cpu)  # 归一化

        X_train_tensor = torch.tensor(X_true_std).float().to(self.device)
        M_train_tensor = get_mask(X_train_tensor)  # 获取掩码矩阵
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, M_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

        # 初始化GAIN模型
        stopper = EarlyStopper(patience=2, min_delta=0.001)
        model = GAIN(train_loader=train_loader)

        optimizer_G = torch.optim.Adam(model.G.parameters())
        optimizer_D = torch.optim.Adam(model.D.parameters())
        model.set_optimizer(optimizer=optimizer_G, generator=True)
        model.set_optimizer(optimizer=optimizer_D, generator=False)

        model.to(self.device)
        model.train(n_epoches=100, verbose=True, stopper=stopper)

        # 填补缺失数据
        X_std = scaler.fit_transform(data_cpu)
        X_std = torch.from_numpy(X_std).to(self.device)
        x_imputed = model.imputation(x=X_std, m=self.mask)
        x_imputed = scaler.inverse_transform(x_imputed.cpu().numpy())  # 反归一化
        return x_imputed
'''

    def impute_once(self, data):
        """

        """
        # 如果数据在GPU上，将其移至CPU并转换为NumPy数组
        data_cpu = data.cpu().numpy() if data.is_cuda else data.numpy()

        # 归一化数据
        scaler = MinMaxScaler()
        X_true_std = scaler.fit_transform(data_cpu)  # 归一化

        X_std = scaler.torch.from_numpy(X_true_std).to(self.device)
        M_std = get_mask(X_std)
        train_dataset = torch.utils.data.TensorDataset(X_std, M_std)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

        model = GAIN(train_loader=train_loader)
        model.to(self.device)
        model.eval()

        # 填补缺失数据
        x_imputed = model.imputation(x=X_std, m=self.mask)
        x_imputed = scaler.inverse_transform(x_imputed.cpu().numpy())  # 反归一化
        return x_imputed
'''
"""
    Example of using GAIN and EarlyStopping classes for data imputation
"""

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from dataload.io import load_pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from gain import GAIN
from utilsgain import get_mask, EarlyStopper
from dataload.io import write_pickle
import matplotlib.pyplot as plt

def add_missings(x: np.array, miss_rate: float) -> np.array:#产生缺失
    """
    Randomly adds NaN values to the data matrix

    @param x:         the original data matrix
    @param miss_rate: proportion of generated missings
    @return:          matrix x with NaN values
    """
    x_missed = np.copy(x).astype(float)
    n, m = x.shape

    for j in range(m):
        nan_indexes = np.random.choice(n, int(n * miss_rate), replace=False)
        x_missed[nan_indexes, j] = np.nan

    return x_missed


if __name__ == '__main__':

    B_bin, B, X_true, Omega, X, mask = load_pickle('/root/autodl-tmp/OTM-CPU/dataset/MIM-ER1.pickle')
    print(X_true)

    print(X)


    SEED = 13
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    MISS_RATE = 0.1
    TRAIN_SIZE = 0.8

    X = torch.from_numpy(X).to(DEVICE)
    # X_true = torch.from_numpy(X_true).to(DEVICE)
    mask = torch.from_numpy(mask).to(DEVICE)
    mask = mask.float()
    #print(mask)
    np.random.seed(SEED)

    # Load data and add missings
    #data = pd.read_csv('letter.csv').values
    X_true_missed = add_missings(X_true, miss_rate=MISS_RATE)



    # Normalization
    scaler = MinMaxScaler()
    X_true_std = scaler.fit_transform(X_true_missed)##归一化

    # train\test split设置训练集和测试集合 8：2
    ##train_cutoff = int(X_true_std.shape[0] * TRAIN_SIZE)
    ##X_train, X_test = X_true_std[:train_cutoff], X_true_std[train_cutoff:]
    X_train= X_true_std
    ##X_actual = X_true[train_cutoff:]

    X_train_tensor = torch.tensor(X_train).float()
    M_train_tensor = get_mask(X_train_tensor)##掩码矩阵
    train_dataset = TensorDataset(X_train_tensor, M_train_tensor)##训练集打包
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    '''
    X_test_tensor = torch.tensor(X_test).float()
    M_test_tensor = get_mask(X_test_tensor)
    X_actual_tensor = torch.tensor(X_actual).float()
    test_dataset = TensorDataset(X_test_tensor, M_test_tensor, X_actual_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)##测试集打包
`   '''
    # Model initialization and training
    stopper = EarlyStopper(patience=2, min_delta=0.001)##早停器，patience=2表示如果在连续 2 次训练中验证集误差没有显著降低，就停止训练；min_delta表示最小误差变化阈值
    model = GAIN(train_loader=train_loader, seed=SEED)

    optimizer_G = torch.optim.Adam(model.G.parameters())
    optimizer_D = torch.optim.Adam(model.D.parameters())
    model.set_optimizer(optimizer=optimizer_G, generator=True)
    model.set_optimizer(optimizer=optimizer_D, generator=False)

    model.to(DEVICE)
    model.train(n_epoches=100, verbose=True, stopper=stopper)


    ##使用训练后的GAIN进行填补，注意训练时数据是归一化的
    X_std = scaler.fit_transform(X.cpu().numpy())
    X_std = torch.from_numpy(X_std).to(DEVICE)
    x_imputed = model.imputation(x=X_std, m=mask)
    x_imputed = scaler.inverse_transform(x_imputed.cpu().numpy())
    print(x_imputed)
    rmse = np.sqrt(mean_squared_error(y_true=X_true, y_pred=x_imputed))
    print(rmse)
    ###填补实验第一次：填补时没有归一化X，填补后和真实数据对比的rmse时0.4986
    ###填补实验第二次：填补时归一化X，填补后和真实数据对比的rmse时0.4775 差别不大
    #X_true=x_imputed
    #package = (B_bin, B, X_true, Omega, X, mask)
    #write_pickle(package, 'D:\pycharmproject\GAIN-pytorch\dataset9_impute_torchgain.pickle')




    '''
    # Model evaluation
    rmse_batch = []

    for x_test_batch, m_batch, x_actual_batch in test_loader:
        x_batch_imputed = model.imputation(x=x_test_batch, m=m_batch)##填补也是在归一化时进行的，填完再反归一化
        x_batch_imputed = scaler.inverse_transform(x_batch_imputed.cpu().numpy())##将归一化后的数据恢复到原始范围
        print(x_batch_imputed)
        rmse = np.sqrt(mean_squared_error(y_true=x_actual_batch.numpy(), y_pred=x_batch_imputed))
        rmse_batch.append(rmse)

    print(f'mean rmse: {np.mean(rmse_batch):.4f}\nstd of rmse: {np.std(rmse_batch):.4f}')

    # Plot train curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(model.history.mse_train, label='train')
    ax.plot(model.history.mse_test, label='test')

    ax.set_xlabel('epoch', fontsize=12, labelpad=10)
    ax.set_ylabel('mean squared error', fontsize=12, labelpad=10)

    ax.legend(fontsize=12)
    plt.show()
    '''
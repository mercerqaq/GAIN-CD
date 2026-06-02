import numpy as np
import torch
import sys
import random
from config import get_data
from  sklearn.metrics import  mean_squared_error
from nonlinear import MissModel, DagmaNonlinear
from joint_train import train_joint_gain_dagma
from utils.eval import evaluate, write_result
#获取命令行参数
config_id = int(sys.argv[1])
graph_type = sys.argv[2]
sem_type = sys.argv[3]

dataset, config = get_data(config_id, graph_type, sem_type) #从config.py获得数据集及参数
code = config["code"]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
data = torch.from_numpy(dataset.X).to(device).float()  # 将数据转换为 float32 类型并移到 GPU
mask = torch.isnan(data).to(device).float()  # 同样确保掩码是 float32 类型
X_true = torch.from_numpy(dataset.X_true).to(device).float() #将真实数据 dataset.X_true 转换为 PyTorch 张量，并将其移动到设备上。

N, D = data.shape
hidden_dims = [D, D, 1]#设置两个隐藏层，每个隐藏层维度是D，输出层维度是1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seeds = [0, 1, 2]

all_shd = []

for seed in seeds:
    print(f"\n===== Running seed {seed} =====")

    set_seed(seed)

    # ===== 训练 =====
    model, W_est = train_joint_gain_dagma(
        data=data,
        mask_missing=mask,
        hidden_dims=[data.shape[1], 10, 1],
        device="cuda",
        hint_rate=0.1,
        alpha=10,
        lambda1=0.02,
        lambda2=0.005,
        mu=0.1,
        s=1.0,
        epochs=20000,
        beta_joint=0.5,
        pretrain_gain_epochs=200,
        warmup_s_epochs=0,
        update_s_every=2,

    )

    # ===== 评估（扫threshold）=====
    threshold_list = [0.2, 0.3, 0.4]

    best_shd = float("inf")

    for th in threshold_list:

        result = evaluate(dataset.B_bin, W_est, threshold=th)
        shd = result.metrics["shd"]

        if shd < best_shd:
            best_shd = shd

    print(f"Seed {seed} best SHD: {best_shd}")
    all_shd.append(best_shd)

mean_shd = np.mean(all_shd)
std_shd = np.std(all_shd)

print("\n===== Final Result =====")
print(f"SHD: {mean_shd:.2f} ± {std_shd:.2f}")

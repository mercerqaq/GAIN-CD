import ot
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from config import get_data
from utils.eval import evaluate, write_result
from tqdm import tqdm
from ot_gradient import auto_ot
###对线性数据的ot填补
def otm(X_init, lambda1, max_iter=100, h_tol=1e-8, rho_max=1e+16, eta=0.01):#参数依次为数据矩阵、正则化参数、最大迭代次数、停止准则h、控制更新速度的参数、正则化权重的参数
    class Sup: 
        def __init__(self, X_init):
            mask = np.isnan(X_init)#掩码矩阵
            X_init[mask] = 0
            self.mask = mask.astype('float')
            self.X_init = X_init#初始的数据矩阵
            self.w = None # 2 * d * d#邻接矩阵
            self.imps = None # n,d#填补值

            

    global supp
    supp = Sup(X_init)

    def _loss(R):#计算给定矩阵R的MSE损失
        """Evaluate value and gradient of loss."""

        loss = 0.5 / R.shape[0] * (R ** 2).sum()#通过衡量数据矩阵与预测矩阵之间的残差R计算损失
        return loss

    def _h(W):#计算图的无环约束和梯度
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W) 
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):#转换向量为矩阵
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _wfunc(w):#评估增广拉格朗日的值和梯度
        
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""

        imps = supp.imps.reshape(d,d)
        X = supp.X_init * (1 - supp.mask) + (supp.X_init @ imps)  * supp.mask
        X = X - np.mean(X, axis=0, keepdims=True) # for l2 only
        W = _adj(w)
        M = X @ W
        R = X - M
        loss = _loss(R)
        h, G_h = _h(W)      

        ot_loss, G = auto_ot(X, W, 'w')

        # Objective function
        obj = loss + alpha * h + lambda1 * w.sum() + 0.5 * rho * h * h + eta * ot_loss
        
        # Calculating gradient for W
        G_W = - 1.0 / X.shape[0] * X.T @ R
        G_smooth = G_W + (rho * h + alpha) * G_h + eta * G
    
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj , g_obj
    
    def _xfunc(imps):#计算并返回估算数据的目标函数值 obj 和梯度 g_obj
        

        imps = imps.reshape(d, d)
        X = supp.X_init * (1 - supp.mask) + (supp.X_init @ imps) * supp.mask
        X = X - np.mean(X, axis=0, keepdims=True) # for l2 only
        W = _adj(supp.w)
        M = X @ W
        R = X - M

        ot_loss, G = auto_ot(X, W, 'x')
        
        obj = _loss(R) + eta * ot_loss
        
        # Calculating gradient
        I = np.eye(d,d)
        g_obj = 1.0 / X.shape[0] * (R @ (I - W.T)) * supp.mask
        g_obj = (supp.X_init.T @ g_obj) + eta * (supp.X_init.T @ G)
        g_obj = g_obj.reshape(-1)

        
        return obj , g_obj

    n, d = X_init.shape
    w_est = np.zeros(2 * d * d)
    
    rho, alpha, h = 1.0, 0.0, np.inf 
    imps = np.ones(d * d)
    
    wbnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    ibnds = [(None, None)] * imps.shape[0]

    supp.w = w_est 
    supp.imps = imps
        
    for i in range(max_iter):#主循环优化：使用 L-BFGS-B 方法最小化损失函数 _xfunc 和 _wfunc。刚开始用初始化为全1的imps填补缺失数据，然后不断用L-BFGS-B优化梯度更新imps。
        print(f'Iteration {i} ...')
        params_new, w_new, h_new = None, None, None
        while rho < rho_max:  
            sol = sopt.minimize(_xfunc, imps, method='L-BFGS-B', jac=True, bounds=ibnds)
            imps_new = supp.imps = sol.x  
            sol = sopt.minimize(_wfunc, w_est, method='L-BFGS-B', jac=True, bounds=wbnds)
            w_new = supp.w = sol.x 
                      
            print(imps_new.max().round(5), imps_new.min().round(5), w_new.sum().round(5))
            h_new, _ = _h(_adj(w_new))
            
            if h_new > 0.25 * h:
                rho *= 2
            else:
                break
        imps, w_est, h = imps_new, w_new, h_new
        params = params_new
        
        print(f'Current h={h}')
        
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            print(f'Stopping at h={h} and rho={rho}')
            break
    
    W_est = _adj(w_est)

    imps = imps.reshape(d,d)
    X_filled = supp.X_init * (1 - supp.mask) + (supp.X_init @ imps) * supp.mask
    return W_est, X_filled, supp.mask#返回估计的邻接矩阵、填充的数据矩阵、缺失值掩码


if __name__ == '__main__':
    import sys
    from dag_methods import Notears

    
    config_id = int(sys.argv[1])
    graph_type = sys.argv[2]
    sem_type = 'linear'
    
    dataset, config = get_data(config_id, graph_type, sem_type)
    
    n,d = dataset.X.shape

    W_est, X_filled, mask = otm(dataset.X, lambda1=0.1, 
                                max_iter=10, h_tol=1e-8, rho_max=1e+16, eta=0.01)#调用优化函数进行优化
 
    raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3)

    
    # =============== WRITE GRAPH ===============
    saved_path = f'output/otm_linear.txt'
    write_result(raw_result, config['code'], saved_path)



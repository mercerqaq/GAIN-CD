import logging
import igraph as ig
import networkx as nx
import numpy as np
from scipy.special import expit as sigmoid

from utils.data import produce_NA
from utils.eval import is_dag
import os
from utils.io import load_pickle, write_pickle
from sklearn.preprocessing import scale 
##模拟数据产生

class SyntheticDataset:#基于DAG生成模拟数据，生成数据矩阵X，权重临界矩阵B和二值邻接矩阵B_bin
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, config_code, graph_type, degree, noise_type,
                 miss_type='mcar', miss_percent=0.1, sem_type='linear',
                 equal_variances=True, mnar_type="logistic", p_obs=0.3, mnar_quantile_q=0.3):#初始化实例的属性
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            graph_type ('ER' or 'SF'): Type of graph.
            degree (int): Degree of graph.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            B_scale (float): Scaling factor for range of B.
            miss_percent (float): Percentage of missing data.
        """
        self.n = n
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.noise_type = noise_type
        self.miss_percent = miss_percent
        self.miss_type = miss_type
        self.sem_type = sem_type
        self.mnar_type = mnar_type
        self.p_obs = p_obs#观察比例，用于模拟MNAR
        self.equal_variances = equal_variances
        self.mnar_quantile_q = mnar_quantile_q#MNAR的参数
        self.B_ranges = ((-2.0, -0.5), (0.5, 2.0))#加权邻接矩阵的权重范围
        self.data_path = f'./dataset/{config_code}.pickle'

        self._setup()
        self._logger.debug("Finished setting up dataset class.")

    def _setup(self):#数据生成，实例也获得了属性，包括邻接矩阵B_bin，加权邻接矩阵B，数据矩阵X_true，噪声方差矩阵Omega，带缺失的数据矩阵X，缺失掩码矩阵mask
        """Generate B_bin, B and X."""
        if os.path.isfile(self.data_path):#路径下如果有数据则加载
            print('Loading data ...')
            self.B_bin, self.B, self.X_true, self.Omega, self.X, self.mask = load_pickle(self.data_path)##加载生成的数据集
        else:
            print('Generating and Saving data ...')
            self.B_bin = SyntheticDataset.simulate_random_dag(self.d,
                                                            self.degree,
                                                            self.graph_type)#生成一个随机DAG的邻接矩阵B_bin

            self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges)#对DAG的边赋予权重，生成加权邻接矩阵B

            if self.sem_type == 'linear':#如果sem是linear则生成该结构方程模型的数据，得到了数据矩阵X_true和噪声方差矩阵Omega
                self.X_true, self.Omega = SyntheticDataset.simulate_linear_sem(
                    self.B, self.n, self.noise_type, self.equal_variances)
            else:#其他sem类型则用非线性方法生成数据，返回值同上
                self.X_true, self.Omega = SyntheticDataset.simulate_nonlinear_sem(
                    self.B_bin, self.n, self.sem_type, self.equal_variances)
            assert is_dag(self.B)

            self.X, self.mask = produce_NA(self.X_true.copy(), p_miss=self.miss_percent, mecha=self.miss_type,
                                        opt=self.mnar_type, p_obs=self.p_obs, q=self.mnar_quantile_q)
            #使用produce_NA 方法根据给定的缺失数据类型（miss_type）、缺失比例（miss_percent）以及其他参数，生成缺失数据集 X 和相应的缺失掩码 mask。
            print(self.B_bin)
            print(self.B)
            print(self.X_true)
            print(self.X)
            print(self.mask)
            package = (self.B_bin, self.B, self.X_true, self.Omega, self.X, self.mask)#邻接矩阵B_bin，加权邻接矩阵B，真实数据矩阵X_true，噪声矩阵Omega，包含缺失的矩阵X，缺失掩码矩阵mask
            write_pickle(package, self.data_path)#生成的数据保存到pickle文件中

    @staticmethod
    def simulate_er_dag(d, degree):#按给定的节点数d和平均度数degree生成一个ER类型的DAG
        """生成ER图的DAG er

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            # return nx.to_numpy_matrix(G)
            return nx.adjacency_matrix(G).todense()##########################由于networkx没有tonumpyarray所以把nx.to_numpy_array(G)改成nx.adjacency_matrix(G).todense()

        p = float(degree) / (d - 1)
        # Probability for edge creation
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    @staticmethod
    def simulate_sf_dag(d, degree):#按给定的节点数d和平均度数degree生成一个sf类型的DAG
        """Simulate ER DAG using igraph package.  sf

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin

    @staticmethod#静态方法，让函数直接成为类的函数，不用再实例化，但是实例化也能调用
    def simulate_random_dag(d, degree, graph_type):#根据er或sf类型生成随机的DAG
        """Simulate random DAG.根据图erorsf生成随机的dag

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            graph_type ('ER' or 'SF'): Type of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = np.random.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        if graph_type == 'ER':
            B_bin = SyntheticDataset.simulate_er_dag(d, degree)
        elif graph_type == 'SF':
            B_bin = SyntheticDataset.simulate_sf_dag(d, degree)
        else:
            raise ValueError("Unknown graph type.")
        return _random_permutation(B_bin)

    @staticmethod
    def simulate_weight(B_bin, B_ranges):#根据输入的邻接矩阵 B_bin 和权重范围 B_ranges 生成’加权的邻接矩阵 B‘。该方法的目的是为图中的每条边分配一个权重，并返回加权后的邻接矩阵。
        """Simulate the weights of B_bin.根据邻接矩阵B_bin和权重范围生成权重返回加权邻接矩阵B

        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        S = np.random.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B

    @staticmethod
    def simulate_linear_sem(B, n, noise_type, equal_variances):
        """Simulate samples from linear SEM with specified type of noise.生成线性sem的数据

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            n (int): Number of samples.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            equal_variances: if the variance is equal.

        Returns:
            numpy.ndarray: [n, d] data matrix.
        """

        def _simulate_single_equation(X, B_i, equal_variances):#根据线性结构方程模型（SEM）模拟第i个节点的样本数据。该方法通过父节点的数据X和权重矩阵B_i 来生成节点的值，并在噪声处理中加入指定类型的噪声。
            """Simulate samples from linear SEM for the i-th node.

            Args:
                X (numpy.ndarray): [n, number of parents] data matrix.
                B_i (numpy.ndarray): [d,] weighted vector for the i-th node.

            Returns:
                numpy.ndarray: [n,] data matrix.
            """
            scale = np.random.uniform(
                low=1.0, high=2.0) if not equal_variances else 1.0
            if noise_type == 'gaussian':
                # Gaussian noise
                N_i = np.random.normal(scale=scale, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = np.random.exponential(scale=scale, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = np.random.gumbel(scale=scale, size=n)
            elif noise_type == 'laplace':
                # Laplace noise
                N_i = np.random.laplace(scale=scale, size=n)
            elif noise_type == 'uniform':
                # Uniform noise
                N_i = np.random.uniform(low=-scale, high=scale, size=n)
            else:
                raise ValueError("Unknown noise type.")
            return X @ B_i + N_i, scale**2 #对于每个节点，使用父节点的样本数据 X[:, parents]和父节点到当前节点的加权系数 B[parents, i]进行线性组合。然后将噪声加到该线性组合结果上。返回模拟的数据 X @ B_i + N_i 和噪声的方差 scale**2。

        d = B.shape[0]
        X = np.zeros([n, d])
        Omega = np.zeros((d, d))    # Noise variance
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i], Omega[i, i] = _simulate_single_equation(
                X[:, parents], B[parents, i], equal_variances)

        return X, Omega.astype(np.float32)

    @staticmethod
    def simulate_nonlinear_sem(B, n, sem_type, equal_variances=True, noise_scale=None):#基于非线性结构方程模型（SEM）模拟数据。它通过给定的邻接矩阵 B 和采样数 n，以及指定的非线性 SEM 类型生成样本数据，并加入噪声。
        """Simulate samples from nonlinear SEM.使用非线性sem生成数据
        Args:
            B (np.ndarray): [d, d] binary adj matrix of DAG
            n (int): num of samples
            sem_type (str): mlp, mim, gp, gp-add
            noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        Returns:
            X (np.ndarray): [n, d] sample matrix
        """

        def _simulate_single_equation(X, scale, equal_variances):
            """X: [n, num of parents], x: [n]"""

            scale = np.random.uniform(
                low=1.0, high=2.0) if not equal_variances else 1.0
            z = np.random.normal(scale=scale, size=n)
            pa_size = X.shape[1]
            if pa_size == 0:
                return z
            if sem_type == 'mlp':#根据SEM类型生成样本，SEM类型可以为MLP、MIM、GP、GP-ADD
                hidden = 100
                W1 = np.random.uniform(
                    low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'mim':
                w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w1[np.random.rand(pa_size) < 0.5] *= -1
                w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w2[np.random.rand(pa_size) < 0.5] *= -1
                w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w3[np.random.rand(pa_size) < 0.5] *= -1
                x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gp-add':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                         for i in range(X.shape[1])]) + z
            else:
                raise ValueError('unknown sem type')
            return x

        d = B.shape[0]
        scale_vec = np.ones(d) if noise_scale is None else noise_scale
        Omega = np.diag(scale_vec**2)
        X = np.zeros([n, d])
        G = ig.Graph.Adjacency(B.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j] = _simulate_single_equation(
                X[:, parents], scale_vec[j], equal_variances)
        return X.astype(np.float32), Omega.astype(np.float32)

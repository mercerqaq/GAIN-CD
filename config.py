from synthetic import SyntheticDataset
#from real import RealDataset


def get_config(config_id, graph_type, sem_type):#根据输入的配置 ID、图类型（graph_type）和 SEM 类型（sem_type）来生成一个字典 config，并设置一些数据集生成的参数。
    
    assert graph_type in ('ER', 'SF', 'REAL'), 'ER, SF, REAL graph only'#图类型，可以是 'ER'（Erdos-Rényi 图）、'SF'（Scale-Free 图）、或 'REAL'（真实世界数据）。

    assert sem_type in ('mlp', 'mim', 'neuro', 'sachs', 'dream1', 'dream2', 'dream3', 'dream4', 'dream5','linear'), 'Incorrect names for SEM type'#SEM（结构方程模型）类型，有不同的选择，如 'mlp'、'mim'、'neuro'、'sachs' 等

    config = {
            'num_obs': 1000,#生成的数据集的样本数为1000
            'num_vars': 20,#数据集的变量数为50
            'graph_type': graph_type,#图类型
            'degree': 2, #图的平均度数为2
            'noise_type': 'gaussian',#高斯噪声
            # 'miss_type': 'mcar',
            # 'miss_percent': 0.1,
            "sem_type": sem_type,#使用的结构方程类型
            "ev": False#不使用相等方差假设
        }

    if config_id in (1,2,3):#####################设置数据的缺失类型和缺失率
        config['miss_type'] = 'mcar'
    elif config_id in (4,5,6):
        config['miss_type'] = 'mar'
    elif config_id in (7,8,9):
        config['miss_type'] = 'mnar'
    
    if config_id in (1,4,7):
        config['miss_percent'] = 0.1
    elif config_id in (2,5,8):
        config['miss_percent'] = 0.3
    if config_id in (3,6,9):
        config['miss_percent'] = 0.5

    
    if config_id > 20: 
        config['miss_type'] = 'mcar'
        config['miss_percent'] = 0.1
    
    if sem_type == 'linear': 
        sem_type = 'Linear'
    else: 
        sem_type = sem_type.upper()#只是把结构方差模型的变量值改成大写
    
    config['code'] = f'{sem_type}-{graph_type}{config_id}'#使用 f-string 格式化创建一个唯一的配置代码，将其存储在 config 字典中
    
    return config


def get_data(config_id, graph_type, sem_type):#用配置字典调整要使用的模拟或真实数据集参数
    
    
    config = get_config(config_id, graph_type, sem_type)#获取配置字典
    
    if graph_type == 'REAL': 
        dataset = RealDataset(n = config['num_obs'], d = config['num_vars'], 
                                config_code = config['code'],
                                miss_type = config['miss_type'], 
                                miss_percent = config['miss_percent'], 
                                sem_type = sem_type,
                                opt="logistic") # replacing mnar opt here e.g., quantile
    else:
        
        dataset = SyntheticDataset(n = config['num_obs'], d = config['num_vars'], 
                            config_code = config['code'],
                            graph_type = config['graph_type'], 
                            degree = config['degree'], 
                            noise_type = config['noise_type'],
                            miss_type = config['miss_type'], 
                            miss_percent = config['miss_percent'], 
                            sem_type = config['sem_type'],
                            equal_variances = config['ev'],
                            mnar_type="logistic" # replacing mnar opt here e.g., quantile
                            )

    return dataset, config

if __name__ == '__main__':

    print('Generating synthetic datasets ...')
    for config_id in range(1,10):#生成模拟数据集
        for graph_type in ('ER', 'SF'):
            for sem_type in ('mlp', 'mim','linear'):
                get_data(config_id, graph_type, sem_type)
    #get_data(1, 'ER', 'linear')

#################由于neuro缺模型，先不做真实数据集上的测试
'''    print('Generating real-world datasets ...')
    for config_id in range(1,10):#生成真实数据集
        for sem_type in ('neuro', 'sachs', 'dream1', 'dream2', 'dream3', 'dream4', 'dream5'):
            get_data(config_id, 'REAL', sem_type)'''
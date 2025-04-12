import numpy as np
import pickle
import os
import urllib.request
import tarfile

def download_cifar10():
    """下载CIFAR-10数据集"""
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    
    if not os.path.exists(filename):
        print('Downloading CIFAR-10 dataset...')
        urllib.request.urlretrieve(url, filename)
        
        print('Extracting files...')
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
    
    print('Dataset downloaded and extracted successfully!')

def load_cifar10_batch(batch_file):
    """加载单个CIFAR-10批次"""
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        X = batch['data']
        y = batch['labels']
        return X, y

def load_cifar10():
    """加载完整的CIFAR-10训练集"""
    download_cifar10()
    
    # 加载训练数据
    X_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = f'cifar-10-batches-py/data_batch_{i}'
        X_batch, y_batch = load_cifar10_batch(batch_file)
        X_train.append(X_batch)
        y_train.extend(y_batch)
    
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    
    return X_train, y_train

def load_cifar10_test():
    """加载CIFAR-10测试集"""
    download_cifar10()
    
    test_file = 'cifar-10-batches-py/test_batch'
    X_test, y_test = load_cifar10_batch(test_file)
    
    return X_test, np.array(y_test)

def preprocess_data(X):
    """
    数据预处理
    :param X: 输入数据
    :return: 预处理后的数据
    """
    # 归一化
    X = X.astype(np.float32) / 255.0
    
    # 减去均值
    mean = np.mean(X, axis=0)
    X -= mean
    
    return X

def one_hot_encode(y, num_classes=10):
    """
    将标签转换为one-hot编码
    :param y: 标签
    :param num_classes: 类别数量
    :return: one-hot编码的标签
    """
    return np.eye(num_classes)[y] 
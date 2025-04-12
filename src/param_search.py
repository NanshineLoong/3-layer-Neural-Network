import numpy as np
from itertools import product
from train import train_model
import json

def parameter_search(X_train, y_train, X_val, y_val):
    """
    执行参数搜索
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param X_val: 验证数据
    :param y_val: 验证标签
    :return: 最佳参数和结果
    """
    # 定义要搜索的参数范围
    hidden_sizes = [64, 128, 256]
    learning_rates = [0.001, 0.01, 0.1]
    l2_regs = [0.001, 0.01, 0.1]
    activations = ['relu', 'sigmoid']
    
    # 存储所有结果
    results = []
    
    # 遍历所有参数组合
    for hidden_size, lr, l2_reg, activation in product(
        hidden_sizes, learning_rates, l2_regs, activations
    ):
        print(f'\nTesting parameters: hidden_size={hidden_size}, '
              f'learning_rate={lr}, l2_reg={l2_reg}, activation={activation}')
        
        # 训练模型
        history = train_model(
            X_train, y_train, X_val, y_val,
            hidden_size=hidden_size,
            learning_rate=lr,
            l2_reg=l2_reg,
            epochs=50,  # 为了加快搜索速度，使用较少的epochs
            batch_size=64,
            activation=activation
        )
        
        # 记录结果
        result = {
            'hidden_size': hidden_size,
            'learning_rate': lr,
            'l2_reg': l2_reg,
            'activation': activation,
            'final_val_accuracy': history['val_accuracies'][-1],
            'final_val_loss': history['val_losses'][-1]
        }
        results.append(result)
        
        # 保存结果到文件
        with open('results/param_search_results.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    # 找到最佳参数
    best_result = max(results, key=lambda x: x['final_val_accuracy'])
    print('\nBest parameters:')
    print(f'Hidden size: {best_result["hidden_size"]}')
    print(f'Learning rate: {best_result["learning_rate"]}')
    print(f'L2 regularization: {best_result["l2_reg"]}')
    print(f'Activation: {best_result["activation"]}')
    print(f'Validation accuracy: {best_result["final_val_accuracy"]:.4f}')
    
    return best_result

if __name__ == '__main__':
    # 加载数据
    from train import load_cifar10
    X, y = load_cifar10()
    
    # 数据预处理
    X = X.reshape(X.shape[0], -1) / 255.0
    y = np.eye(10)[y]
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 执行参数搜索
    best_params = parameter_search(X_train, y_train, X_val, y_val) 
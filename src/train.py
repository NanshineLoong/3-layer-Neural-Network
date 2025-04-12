import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import NeuralNetwork
from utils import load_cifar10, preprocess_data, one_hot_encode

def train_model(X_train, y_train, X_val, y_val, hidden_size=128, 
                learning_rate=0.01, l2_reg=0.01, epochs=100, 
                batch_size=64, activation='relu'):
    """
    训练模型
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param X_val: 验证数据
    :param y_val: 验证标签
    :param hidden_size: 隐藏层大小
    :param learning_rate: 学习率
    :param l2_reg: L2正则化强度
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param activation: 激活函数类型
    :return: 训练历史
    """
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    model = NeuralNetwork(input_size, hidden_size, output_size, activation)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    
    for epoch in tqdm(range(epochs)):
        # 学习率衰减
        current_lr = learning_rate * (0.95 ** (epoch // 10))
        
        # 随机打乱数据
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # 前向传播
            output = model.forward(X_batch)
            
            # 反向传播
            loss = model.backward(X_batch, y_batch, output, current_lr, l2_reg)
            epoch_loss += loss * len(X_batch)
        
        # 计算平均损失
        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)
        
        # 验证
        val_output = model.forward(X_val)
        val_loss = -np.sum(y_val * np.log(val_output) + (1 - y_val) * np.log(1 - val_output)) / len(X_val)
        val_loss += (l2_reg / (2 * len(X_val))) * (np.sum(model.W1**2) + np.sum(model.W2**2))
        val_losses.append(val_loss)
        
        # 计算验证集准确率
        val_predictions = model.predict(X_val)
        val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))
        val_accuracies.append(val_accuracy)
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_weights('models/best_model.npz')
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('results/training_curves.png')
    plt.close()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

if __name__ == '__main__':
    # 加载数据
    X, y = load_cifar10()
    
    # 数据预处理
    X = preprocess_data(X)
    y = one_hot_encode(y)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    history = train_model(
        X_train, y_train, X_val, y_val,
        hidden_size=128,
        learning_rate=0.01,
        l2_reg=0.01,
        epochs=100,
        batch_size=64,
        activation='relu'
    ) 
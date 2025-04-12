import numpy as np
from model import NeuralNetwork
import matplotlib.pyplot as plt
from utils import load_cifar10_test, preprocess_data

def visualize_weights(model):
    """可视化模型权重"""
    plt.figure(figsize=(15, 5))
    
    # 可视化第一层权重
    plt.subplot(1, 2, 1)
    plt.imshow(model.W1.T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('First Layer Weights')
    
    # 可视化第二层权重
    plt.subplot(1, 2, 2)
    plt.imshow(model.W2.T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Second Layer Weights')
    
    plt.savefig('results/weights_visualization.png')
    plt.close()

def test_model(model_path, X_test, y_test):
    """
    测试模型性能
    :param model_path: 模型权重文件路径
    :param X_test: 测试数据
    :param y_test: 测试标签
    """
    # 创建模型实例
    input_size = X_test.shape[1]
    hidden_size = 128  # 需要与训练时使用的隐藏层大小相同
    output_size = 10   # CIFAR-10有10个类别
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    # 加载模型权重
    model.load_weights(model_path)
    
    # 在测试集上进行预测
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # 可视化模型权重
    visualize_weights(model)
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(10):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(predictions[class_mask] == y_test[class_mask])
            class_accuracies.append(class_accuracy)
            print(f'Class {i} Accuracy: {class_accuracy:.4f}')
    
    # 绘制类别准确率柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(range(10), class_accuracies)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Class')
    plt.savefig('results/class_accuracies.png')
    plt.close()

if __name__ == '__main__':
    # 加载测试数据
    X_test, y_test = load_cifar10_test()
    
    # 数据预处理
    X_test = preprocess_data(X_test)
    
    # 测试模型
    test_model('models/best_model.npz', X_test, y_test) 
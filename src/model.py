import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """
        初始化三层神经网络
        :param input_size: 输入层大小
        :param hidden_size: 隐藏层大小
        :param output_size: 输出层大小
        :param activation: 激活函数类型，支持 'relu' 和 'sigmoid'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        """
        前向传播
        :param X: 输入数据
        :return: 输出层的激活值
        """
        # 第一层
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.activation == 'relu':
            self.a1 = self._relu(self.z1)
        else:
            self.a1 = self._sigmoid(self.z1)
            
        # 第二层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._sigmoid(self.z2)  # 输出层使用sigmoid
        
        return self.a2
    
    def backward(self, X, y, output, learning_rate, l2_reg):
        """
        反向传播
        :param X: 输入数据
        :param y: 真实标签
        :param output: 前向传播的输出
        :param learning_rate: 学习率
        :param l2_reg: L2正则化强度
        :return: 损失值
        """
        m = X.shape[0]
        
        # 计算损失
        loss = -np.sum(y * np.log(output) + (1 - y) * np.log(1 - output)) / m
        loss += (l2_reg / (2 * m)) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        
        # 计算输出层误差
        dZ2 = output - y
        dW2 = np.dot(self.a1.T, dZ2) / m + (l2_reg / m) * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # 计算隐藏层误差
        if self.activation == 'relu':
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self._relu_derivative(self.z1)
        else:
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self._sigmoid_derivative(self.a1)
            
        dW1 = np.dot(X.T, dZ1) / m + (l2_reg / m) * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # 更新参数
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
        return loss
    
    def predict(self, X):
        """
        预测
        :param X: 输入数据
        :return: 预测的类别
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def save_weights(self, path):
        """
        保存模型权重
        :param path: 保存路径
        """
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    
    def load_weights(self, path):
        """
        加载模型权重
        :param path: 权重文件路径
        """
        weights = np.load(path)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2'] 
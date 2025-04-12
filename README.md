# 三层神经网络分类器

这是一个使用NumPy实现的三层神经网络分类器，用于CIFAR-10图像分类任务。

## 项目结构

```
neural_network_classifier/
├── src/
│   ├── model.py      # 神经网络模型实现
│   ├── train.py      # 训练脚本
│   ├── test.py       # 测试脚本
│   ├── param_search.py # 参数搜索脚本
│   └── utils.py      # 工具函数
├── data/             # 数据集目录
├── models/           # 模型权重保存目录
├── results/          # 结果保存目录
└── requirements.txt  # 项目依赖
```

## 环境要求

- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn
- tqdm

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python src/train.py
```

### 2. 测试模型

```bash
python src/test.py
```

### 3. 参数搜索

```bash
python src/param_search.py
```

## 模型说明

- 输入层：3072个神经元（32x32x3图像）
- 隐藏层：可配置大小（默认128）
- 输出层：10个神经元（对应CIFAR-10的10个类别）
- 激活函数：支持ReLU和Sigmoid
- 优化器：SGD（随机梯度下降）
- 损失函数：交叉熵损失
- 正则化：L2正则化

## 实验结果

训练过程中会生成以下可视化结果：
1. 训练和验证损失曲线
2. 验证集准确率曲线
3. 模型权重可视化
4. 每个类别的准确率柱状图

## 注意事项

1. 首次运行时会自动下载CIFAR-10数据集
2. 训练好的模型权重会保存在`models/best_model.npz`
3. 所有实验结果会保存在`results/`目录下

## 许可证

MIT License 
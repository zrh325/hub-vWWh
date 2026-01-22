"""
调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

X_numpy = np.random.uniform(-2*np.pi, 2*np.pi, 1000)
Y_numpy = 5*np.sin(X_numpy) + np.random.normal(0 ,0.1, 1000)
x = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(Y_numpy).float()

learn_rate = 0.01

# 定义多层神经网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1000, 500),  # 输入层到隐藏层1
            nn.ReLU(),  # 激活函数
            nn.Linear(500, 250),  # 隐藏层1到隐藏层2
            nn.ReLU(),  # 激活函数
            nn.Linear(250, 500),  # 隐藏层2到隐藏层3
            nn.ReLU(),  # 激活函数
            nn.Linear(500, 1000)  # 隐藏层3到输出层
        )

    def forward(self, x):
        return self.network(x)

model = MLP()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), learn_rate)
epochs = 1000
for epoch in range(epochs):
    y_predicted = model(x)

    #计算损失
    loss = loss_fn(y_predicted, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    if (epoch + 1) % 1 == 0:
        print(f'当前为第{epoch + 1}次,  计算损失为{loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(x)

# 对X进行排序，并获取对应的y值
sorted_indices = np.argsort(X_numpy)  # 获取排序索引
X_sorted = X_numpy[sorted_indices]
y_pred_sorted = y_predicted.numpy()[sorted_indices]
y_true_sorted = Y_numpy[sorted_indices]

# 绘制平滑的曲线（按X从小到大连接）
plt.plot(X_sorted, y_pred_sorted, label=f'predicted_model: y = 5sin(x)',
         color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
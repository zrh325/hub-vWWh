import torch
import numpy as np
import matplotlib.pyplot as plt

# ===================== 字体设置 =====================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成 sin(x) 数据
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)   # 等间距采样
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1) # 加噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin 数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络
model = torch.nn.Sequential(
    torch.nn.Linear(1, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1)
)

# 3. 损失函数与优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print("---" * 10)

# 5. 绘图
with torch.no_grad():
    y_pred_plot = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw Data (with noise)', color='blue', alpha=0.6)
plt.plot(X_numpy, np.sin(X_numpy), label='True: y = sin(x)', color='green', linewidth=2)
plt.plot(X_numpy, y_pred_plot, label='Neural Network Fit', color='red', linewidth=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Multi-Layer Neural Network Fitting sin(x)')
plt.legend()
plt.grid(True)
plt.show()
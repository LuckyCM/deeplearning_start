# 用深度学习框架来简洁地实现线性回归模型

#################################
# 3.3.1. 生成数据集
#################################
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # """生成y=Xw+b+ϵ"""

#################################
# 3.3.2. 读取数据集
#################################
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构建一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)   # is_train每个周期都打乱数据

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

#################################
# 3.3.3. 定义模型
#################################
# 第一个指定输入特征形状，即 2，第二个指定输出特征形状
from torch import nn  # 神经网络nn
net = nn.Sequential(nn.Linear(2, 1))

#################################
# 3.3.4. 初始化模型参数
#################################
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

#################################
# 3.3.5. 定义损失函数(计算均方误差使用的是MSELoss类) L2
#################################
loss = nn.MSELoss()

#################################
# 3.3.6. 定义优化算法
#################################
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#################################
# 3.3.7. 训练
#################################
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch{epoch + 1}, loss{l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
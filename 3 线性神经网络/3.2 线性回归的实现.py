import random
import torch
from d2l import torch as d2l
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#################################
# 3.2.1. 生成数据集
#################################
# 生成一个包含1000个样本的数据集
# y=Xw+b+ϵ
def synthetic_data(w, b, num_examples): #@save
    """生成y=Xw+b+ϵ"""
    X = torch.normal(0, 1, (num_examples, len(w))) # normal(mean, std, size)
    # print(X) # 正态分布X
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)    # 噪声的均值=0, 标准差=0.01
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

# 通过生成第二个特征 features[:, 1] 和 labels 的散点图，可以直观地观察到两者之间的线性关系
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1) # detach()不需要计算梯度
d2l.plt.show()

#################################
# 3.2.2. 读取数据集
#################################
# 抽取一小批量样本
def data_iter(batch_size, feature, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield feature[batch_indices], labels[batch_indices]  # yield是一个生成器

# 用GPU处理并行数据
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

#################################
# 3.2.3. 初始化模型参数
#################################
import numpy as np
# 从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print(w, b)

#################################
# 3.2.4. 定义模型
#################################
def linreg(X, w, b):  #@save
    """线性回归模型y=wX+b"""
    return torch.matmul(X, w) + b

#################################
# 3.2.5. 定义损失函数
#################################
def squared_loss(y_hat, y):  #@save
    """均方损失f(x) = (y^-y)^2 /2"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

#################################
# 3.2.6. 定义优化算法
#################################
def sgd(params, lr, batch_size):    #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#################################
# 3.2.7. 训练
#################################
# 概括一下，我们将执行以下循环：
# 1. 初始化参数
# 重复，直到完成:
#   2. 计算梯度
#   3. 更新参数
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')  # 标量 不用reshape



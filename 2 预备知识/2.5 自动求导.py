#################################
# 2.5.1 一个简单的例子
#################################
import torch
from matplotlib.ticker import FuncFormatter

x = torch.arange(4.0)
print(x)

# 存储梯度
x.requires_grad_(True)  # 等价于 `x = torch.arange(4.0, requires_grad=True)`
print(x.grad) # 默认值是None

# 计算y  (y=2x⊤x)
y = 2 * torch.dot(x, x)
print(y)

# 通过调用反向传播函数来自动计算y关于x 每个分量的梯度
y.backward()
print(x.grad)
print(x.grad == 4 * x)

# 另一个函数例子
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()  # 清除梯度
y = x.sum()
print(y)
y.backward()
print(x.grad)

#################################
# 2.5.2 非标量变量的反向传播
#################################
# 对非标量调用`backward`需要传入一个`gradient`参数，该参数指定微分函数关于`self`的梯度
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
print(y)
y.sum().backward()  # 等价于y.backward(torch.ones(len(x)))
print(x.grad)

#################################
# 2.5.3 分离计算 z = u * x
#################################
x.grad.zero_()
y = x * x
u = y.detach()  # 分离y
z = u * x
print(y, u, z)
z.sum().backward()  # 对z求x的偏导
print(x.grad == u)

# y 上调用反向传播
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

#################################
# 2.5.4 Python控制流的梯度计算
#################################
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)    # a只能是标量
d.backward()
print(a, d)
print(a.grad == d / a)

#################################
# exercise 2
#################################
# 要进行两次反向传播y.backward(retain_graph=True)使他中间值进行保存
import torch
x = torch.arange(40.,requires_grad=True)
y = 2 * torch.dot(x**2,torch.ones_like(x))
y.backward(retain_graph=True)
x.grad
y.backward()

#################################
# exercise 3
#################################
a = torch.randn(size=(3,1), requires_grad=True) # a只能是标量
print(a.shape)
print(a)
d = f(a)
# d.backward() #<====== run time error if a is vector or matrix RuntimeError: grad can be implicitly created only for scalar outputs
d.sum().backward() #<===== this way it will work
print(d, d.sum())

#################################
# exercise 4
#################################
from d2l import torch as d2l
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

fig, ax = d2l.plt.subplots(1)

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
x = np.arange(-2*np.pi, 2*np.pi, 0.025*np.pi)
print(x)
x1 = torch.tensor(x, requires_grad=True)    # 转x为tensor
y1 = torch.sin(x1)
y1.sum().backward()     # y1是矢量, y1.sum()是标量

ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, x1.grad, label="gradient of sin(x)")
ax.legend(loc='upper center', shadow=True)

# 更改横坐标
ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: '{:g}\pi'.format(val/np.pi) if val !=0 else '0'))
ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

d2l.plt.show()
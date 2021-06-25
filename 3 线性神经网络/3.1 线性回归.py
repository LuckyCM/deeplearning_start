import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#################################
# 3.1.2. 矢量化加速(使用矢量进行数据计算)
#################################
# 同时处理整个小批量的样本，需要我们对计算进行矢量化，从而利用线性代数库，而不是在Python中编写开销高昂的for循环

# 我们实例化两个全1的10000维向量
# 在一种方法中，我们将使用Python的for循环遍历向量
# 在另一种方法中，我们将依赖对 + 的调用
n = 10000
a = torch.ones(n)
b = torch.ones(n)

class Timer:    #@save
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

# 1.使用for循环，每次执行一位的加法
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')

# 2.使用重载的 + 运算符来计算按元素的和
timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')

#################################
# 3.1.3. 正态分布与平方损失
#################################
# 定义一个Python函数来计算正态分布
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)

# 可视化正态分布
# 再次使用numpy进行可视化
x = np.arange(-7, 7, 0.01)  # 把这些点都画出来
print(x)
# 三个均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show()


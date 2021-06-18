import torch
from torch.distributions import multinomial  # 多项分布
from d2l import torch as d2l
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#################################
# 2.6.1 基本概率论
#################################
fair_probs = torch.ones([6]) / 6
print(fair_probs)

# 用for速度慢
for i in range(10):
    sample = multinomial.Multinomial(1, fair_probs).sample()  # 随机取样
    print(sample)

# 所以同时采取多个样本
samples = multinomial.Multinomial(10, fair_probs).sample()
print("samples =", samples)

# 计算相对频率作为真实概率的估计
counts = multinomial.Multinomial(1000, fair_probs).sample()
prob = counts / 1000
print("prob =", prob)

# 500组实验，每组抽取10个样本
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)  # 各行的累加值
print(cum_counts)
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)

d2l.set_figsize((6, 4.5))   # 图片长6,宽4.5
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')     # 横虚线
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()

#################################
# 2.3.1 标量
#################################
# 标量由只有一个元素的张量表示
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

print(x + y, x * y, x / y, x**y)

#################################
# 2.3.2 向量
#################################
x = torch.arange(12).reshape((3,4))
print(x)

# print(x[3])

# 张量的长度, 行数
print(len(x))
# dimension
print(x.shape)

#################################
# 2.3.3 矩阵
#################################
A = torch.arange(20).reshape(5, 4)
print(A)
print(A[4, 3])
print(A.T)

B = torch.tensor([[1,2,3], [2,0,4], [3,4,5]])
print(B)

print(B == B.T)

#################################
# 2.3.4 张量
#################################
X = torch.arange(24).reshape(2, 3, 4)
print(X)

#################################
# 2.3.5 张量算法的基本性质(对矩阵运算不会影响矩阵形状) 哈达玛积⊙
#################################
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A+B)
print(A*B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)

#################################
# 2.3.6 降维
#################################
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum)
print(A)
print(A.shape)
print(A.sum())

A_sum_axis0 = A.sum(axis=0)     # 每列的所有值相加(沿轴0)
print(A_sum_axis0)
print(A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)     # 每行的所有值相加(沿轴1)
print(A_sum_axis1)
print(A_sum_axis1.shape)

A_sum_axis01 = A.sum(axis=[0, 1])  # Same as `A.sum()`
print(A_sum_axis01)

# 求平均值
mean1 = A.mean()
mean2 = A.sum() / A.numel()
print(mean1, mean2)

# 同样，计算平均值的函数也可以沿指定轴降低张量的维度
print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

#################################
# 2.3.6.1 非降维求和
#################################
sum_A = A.sum(axis=1, keepdims=True)  # keepdims=True保证了沿轴的方向没有改变
print(sum_A)

# 例如，由于 sum_A 在对每行进行求和后仍保持两个轴，我们可以通过广播将 A 除以 sum_A
print(A/sum_A)  # True
# print(A/A.sum(axis=1))  # False

# 沿某个轴计算 A 元素的累积总和(每一行累加)
cumsum = A.cumsum(axis=0)
print(cumsum)

#################################
# 2.3.7 点积(Dot Product)  x⊤y (1x1)
#################################
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))


#################################
# 2.3.8 矩阵-向量积 (mx1)
#################################
# torch.mul()是矩阵的点乘，即对应的位相乘，要求shape一样, 返回的还是个矩阵
# torch.mm()是矩阵正常的矩阵相乘，（a, b）* ( b, c ) = ( a, c )
# torch.dot()类似于mul()，它是向量(即只能是一维的张量)的对应位相乘再求和，返回一个tensor数值
# torch.mv()是矩阵和向量相乘，类似于torch.mm()

import numpy as np
print(A.shape, x.shape, torch.mv(A, x))
print(np.dot(A, x))     # same
# print(torch.dot(A, x))  # 1x1 不适用
print(torch.mv(A, y.T))

#################################
# 2.3.9 矩阵-矩阵乘法 (mxn * nxk)
#################################
B = torch.ones(4, 3)
print(torch.mm(A, B))

#################################
# 2.3.10 范数
#################################
# L2范式
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))    # L2范式

#  L1范数，它表示为向量元素的绝对值之和
print(torch.abs(u).sum())

# Lp范式(矩阵向量的L2范式) 弗罗贝尼乌斯范数
print(torch.ones((4, 9)))
print(torch.norm(torch.ones((4, 9))))

print(torch.ones((2, 4, 9)))
print(torch.norm(torch.ones((2, 4, 9))))







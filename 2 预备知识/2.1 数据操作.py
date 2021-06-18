import torch

#################################
# 2.1.1 入门
#################################
# 使用 arange 创建一个行向量 x, 这个行向量包含12个元素（默认浮点数）。
x = torch.arange(12)
print(x)

# 通过张量的 shape 属性来访问张量的形状
print(x.shape)

# 张量中元素的总数，即所有元素的形状乘积
print(x.numel())

# 要改变一个张量的形状(不改变元素数量和元素值) (设置-1自动除，当且仅当可行时)
# X = x.reshape(3, 4)
# X = x.reshape(-1, 4)
X = x.reshape(3, -1)
print(X)

# 全0矩阵
zeros = torch.zeros((2, 3, 4)) # 2x3x4矩阵
print(zeros)

# 全1矩阵
ones = torch.ones(2, 3, 4)
print(ones)

# 从均值为0、标准差为1的标准高斯（正态）分布中随机采样(random)
randoms = torch.randn(3, 4)
print(randoms)

# 嵌套列表
tensors = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(tensors)

#################################
# 2.1.2 计算
#################################
# 元素计算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x**y)
print(torch.exp(x))

# 张量连结(concatenate)
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0)) # 沿行(轴-0):竖着6x4
print(torch.cat((X, Y), dim=1)) # 按列(轴-1):横着3x8

# 逻辑运算符(相等的值显示为True,不相等为False)
print(X == Y)

# 对所有元素求和
print(X.sum())

#################################
# 2.1.3 广播机制
#################################
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)

# 广播(矩阵a将复制列，矩阵b将复制行，然后再按元素相加)
print(a + b)

#################################
# 2.1.4 索引和切片
#################################
print(X[-1])  # 打印最后一行
print(X[1: 3])  # 打印从第2行到第3行 [1,3)

X[1, 2] = 9 # 写入数据
print(X)

X[0:2, :] = 12
print(X)

#################################
# 2.1.5 节省内存 (两个方法: 1)X[:] = X + Y; 2)X += Y )
#################################
before = id(Y)
# 分配了新的内存地址
# Y = Y + X  # False
Y[:] = Y + X   # True
print(id(Y) == before)

# 这是不可取的: 1)机器学习数据大 2)没有实时更新的话一些引用可能会指向旧地址
# 解决方法: 使用切片表示法将操作的结果分配给先前分配的数组 (Y[:] = <expression>)
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
print(id(X) == before)

#################################
# 2.1.6 转换为其他 Python 对象
#################################
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))
print(id(A) == id(B))

# 将大小为1的张量转换为 Python 标量
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))




#################################
# 2.2.1 读取数据集
#################################
import os

# 创建一个人工数据集，并储存在../data/house_tiny.csv中
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)

#################################
# 2.2.2 处理缺失值 (插值 和 删除)
#################################
# 只介绍插值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 对于 inputs 中缺少的的数值，我们用同一列的均值替换 “NaN”
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 将类别值或离散值转化为二进制值
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

#################################
# 2.2.3 转换为张量格式
#################################
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)
















# 

print("线性回归的简洁实现")

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# nn是神经网络的缩写
from torch import nn


import math

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)



if __name__ == "__main__":



    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    # 读取数据集
    # 我们可以[调用框架中现有的API来读取数据]。 我们将features和labels作为API的参数传递，并通过数据迭代器指定batch_size。 此外，布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。



    batch_size = 2**10
    data_iter = load_array((features, labels), batch_size)
    # 使用data_iter的方式与我们在 :numref:sec_linear_scratch中使用data_iter函数的方式相同。为了验证是否正常工作，让我们读取并打印第一个小批量样本。 与 :numref:sec_linear_scratch不同，这里我们使用iter构造Python迭代器，并使用next从迭代器中获取第一项。

    a = next(iter(data_iter))

    # print(a)



    net = nn.Sequential(nn.Linear(2, 1))

    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)


    loss = nn.MSELoss()


    trainer = torch.optim.SGD(net.parameters(), lr=0.03)


    num_epochs = 2**8
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X) ,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        ll = -math.log(l)
        ll = int(ll)
        # print('loss',ll)
        if(ll>=9 ):
            print("epoch",epoch)
            break
        # print(f' loss {ll}')
        # print(f'epoch {epoch + 1}, loss {ll:f}')



    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
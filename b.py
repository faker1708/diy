
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 1
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

## 模型

# 因为我们忽略了空间结构，
# 所以我们使用`reshape`将每个二维图像转换为一个长度为`num_inputs`的向量。
# 只需几行代码就可以(**实现我们的模型**)。


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

## 损失函数

# 由于我们已经从零实现过softmax函数（ :numref:`sec_softmax_scratch`），
# 因此在这里我们直接使用高级API中的内置函数来计算softmax和交叉熵损失。
# 回想一下我们之前在 :numref:`subsec_softmax-implementation-revisited`中
# 对这些复杂问题的讨论。
# 我们鼓励感兴趣的读者查看损失函数的源代码，以加深对实现细节的了解。


loss = nn.CrossEntropyLoss(reduction='none')

# ## 训练

# 幸运的是，[**多层感知机的训练过程与softmax回归的训练过程完全相同**]。
# 可以直接调用`d2l`包的`train_ch3`函数（参见 :numref:`sec_softmax_scratch` ），
# 将迭代周期数设置为10，并将学习率设置为0.1.


num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 为了对学习到的模型进行评估，我们将[**在一些测试数据上应用这个模型**]。


d2l.predict_ch3(net, test_iter)

# ## 小结

# * 手动实现一个简单的多层感知机是很容易的。然而如果有大量的层，从零开始实现多层感知机会变得很麻烦（例如，要命名和记录模型的参数）。

# ## 练习

# 1. 在所有其他参数保持不变的情况下，更改超参数`num_hiddens`的值，并查看此超参数的变化对结果有何影响。确定此超参数的最佳值。
# 1. 尝试添加更多的隐藏层，并查看它对结果有何影响。
# 1. 改变学习速率会如何影响结果？保持模型架构和其他超参数（包括轮数）不变，学习率设置为多少会带来最好的结果？
# 1. 通过对所有超参数（学习率、轮数、隐藏层数、每层的隐藏单元数）进行联合优化，可以得到的最佳结果是什么？
# 1. 描述为什么涉及多个超参数更具挑战性。
# 1. 如果想要构建多个超参数的搜索方法，请想出一个聪明的策略。


# [Discussions](https://discuss.d2l.ai/t/1804)

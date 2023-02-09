print("线性回归的从零实现")


# %matplotlib inline
import random
import torch
from d2l import torch as d2l
import math


def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b

    # 均值 ，方差
    average = 0
    aa = 5
    variance = 2**(-aa)

    y += torch.normal(average, variance, y.shape)
    return X, y.reshape((-1, 1))




def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]




def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b



def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2



def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == "__main__":


    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    print('features:', features[0],'\nlabel:', labels[0])


    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);




    # for X, y in data_iter(batch_size, features, labels):
    #     print(X, '\n', y)
    #     break


    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)



    batch_size = 2**4
    lr = 0.03   # 学习率
    num_epochs = 2**10   # 最大训练次数
    net = linreg
    loss = squared_loss     # 定义损失函数


    patience_ratio = 4  # 比如总数是1024 耐心率是 4 则 耐心度为 1024/4 = 256 当精度连续256次都没有升高时，认为孺子不可教，放弃训练
    patience = num_epochs/patience_ratio

    max_precision = 0   # 记录最大精度

    pa_count = 0

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            # if(epoch%16 == 0):

                # print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
            ll = math.log(train_l.mean())
            ll = -int(ll)
            now_precision = ll

            demand_precision = 9  # 对精度的要求

            if(now_precision > max_precision):
                pa_count = 0 # 耐心恢复
                max_precision = now_precision
                print(now_precision)
            else:
                pa_count+=1
                print(pa_count ,end =' ')
            if(pa_count>=patience):
                print('连续训练了',pa_count,'次都没有进展,学习中断,当前精度为',now_precision)
                break
            # print(now_precision,end = ' ')
            if(now_precision>=demand_precision):
                print("达到了目标精度,训练成本为",epoch)
                break


            # print('.',end='')
            # print(train_l.mean())

    print("batch_size",batch_size)
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')





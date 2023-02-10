import torch



def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2



def sgd(param, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    print("小批量随机梯度下降")
    with torch.no_grad():


        gvg =  param.grad / batch_size
        offset = -lr * gvg
        param += offset
        # param -= lr * param.grad / batch_size
        param.grad.zero_()
        w = param
        w[-1,0:-1]=0
        w[-1,-1]=1

        print(w)

if __name__ == "__main__":

    nn = torch

    dim_x = 2+1
    dim_y = 1+1

    dim_w = [4+1]
    lw = list()
    lx = list()


    # x0
    #x = torch.randn(dim_x, 1)
    x = torch.tensor([0.,1.,1.])
    y = torch.tensor([4.,1.])

    x[dim_x-1]=1
    lx.append(x)

    w = torch.randn(dim_w[0], dim_x, requires_grad=True)
    # w[dim_w[0]-1] = torch.zeros()
    with torch.no_grad():
        w[-1,0:-1]=0
        w[-1,-1]=1
    # print(w)
    lw.append(w)

    
    # w1
    w = torch.randn(dim_y,dim_w[0],requires_grad = True)
    with torch.no_grad():
        w[-1,0:-1]=0
        w[-1,-1]=1
    lw.append(w)









    for i in range(0,3):
        #x1
        x = torch.matmul(lw[0],lx[0])
        x = torch.clamp(x,min=0.0)
        # print(x)
        lx.append(x)


        # x2
        x = torch.matmul(lw[1],lx[1])
        x = torch.clamp(x,min=0.0)
        lx.append(x)



        # loss = nn.MSELoss()

        # xn = lx[-1]
        xn = x
        # print('xn',xn)
        loss = squared_loss
        l = loss(xn,y)
        # print("损失 l",l)

        ls = l.sum()
        print(ls)
        lsb = ls.backward()
        # print(lsb)
        # l.sum().backward()
        # print("损失 l",l)

        #lx[-1].sum().backward()

        # c = torch.sum(lx[-1])


        # print("打印w")

        # for i,ele in enumerate(lw):
        #     print(ele)

        # print("打印x")

        # for i,ele in enumerate(lx):
        #     print(ele)

        # print("打印梯度")


        # # c.backward()
        # print(lw[0].grad)
        # print(lw[1].grad)

        lr = 0.1
        bs = 1
        for i in range(len(lw)):
            sgd(lw,lr,bs)
        # w = lw[-1]
        # sgd(w,lr,bs)
        
        # w = lw[-2]
        # sgd(w,lr,bs)


    # a = torch.randn(5)
    # print(a)
    # a = torch.clamp(a,0.,100.)
    # print(a)


    # a = torch.zeros(3)
    # print('a',a)

    # w = torch.randn(3, dim_x, requires_grad=True)
    # w[-1,0:-2] = 0
    # w[-1,-1]=1
    # print(w)


    # n=torch.FloatTensor(3,4).fill_(33)
    # n[-1,0:-1]=0
    # n[-1,-1] = 1
    # print(n)


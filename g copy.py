import torch


dim_x = 2
dim_y = 1

dim_w = [4]
lw = list()

x0 = torch.randn(2, 1)


w = torch.randn(dim_w[0], 2, requires_grad=True)
lw.append(w)

x1 = torch.matmul(lw[0],x0)


w1 = torch.randn(1,4,requires_grad = True)

x2 = torch.matmul(w1,x1)

c = torch.sum(x2)




c.backward()
print(w0.grad)
print(w1.grad)
import torch


dim_x = 2
dim_y = 1

dim_w = [4]
lw = list()
lx = list()

x = torch.randn(dim_x, 1)
lx.append(x)

w = torch.randn(dim_w[0], dim_x, requires_grad=True)
lw.append(w)

x1 = torch.matmul(lw[0],lx[0])

# w1
w = torch.randn(dim_y,dim_w[0],requires_grad = True)
x = torch.matmul(w,x1)

c = torch.sum(x)
lx.append(x)





c.backward()
print(lw[0].grad)
print(lw[1].grad)
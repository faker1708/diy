import torch


dim_x = 2+1
dim_y = 1+1

dim_w = [4+1]
lw = list()
lx = list()

x = torch.randn(dim_x, 1)
x[dim_x-1]=1
lx.append(x)

w = torch.randn(dim_w[0], dim_x, requires_grad=True)
lw.append(w)

x = torch.matmul(lw[0],lx[0])
torch.clamp(x,min=0.0)
print(x)
lx.append(x)

# w1
w = torch.randn(dim_y,dim_w[0],requires_grad = True)
lw.append(w)

x = torch.matmul(w,lx[-1])
torch.clamp(x,min=0.0)
lx.append(x)


c = torch.sum(lx[-1])


for i,ele in enumerate(lx):
    print(ele)



c.backward()
# print(lw[0].grad)
# print(lw[1].grad)


a = torch.randn(5)
print(a)
torch.clamp(a,min=0.0)
print(a)
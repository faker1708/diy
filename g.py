import torch


dim_x = 2+1
dim_y = 1+1

dim_w = [4+1]
lw = list()
lx = list()


# x0
x = torch.randn(dim_x, 1)
x[dim_x-1]=1
lx.append(x)

w = torch.randn(dim_w[0], dim_x, requires_grad=True)
# w[dim_w[0]-1] = torch.zeros()
lw.append(w)

#x1
x = torch.matmul(lw[0],lx[0])
x = torch.clamp(x,min=0.0)
# print(x)
lx.append(x)

# w1
w = torch.randn(dim_y,dim_w[0],requires_grad = True)
lw.append(w)

# x2
x = torch.matmul(w,lx[-1])
x = torch.clamp(x,min=0.0)
lx.append(x)


c = torch.sum(lx[-1])


for i,ele in enumerate(lx):
    print(ele)



c.backward()
# print(lw[0].grad)
# print(lw[1].grad)


# a = torch.randn(5)
# print(a)
# a = torch.clamp(a,0.,100.)
# print(a)


a = torch.zeros(3)
print('a',a)

w = torch.randn(3, dim_x, requires_grad=True)
# w[-1,0:-2] = 0
# w[-1,-1]=1
# print(w)


n=torch.FloatTensor(3,4).fill_(33)
n[-1,0:-1]=0
n[-1,-1] = 1
print(n)
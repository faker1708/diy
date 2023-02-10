import torch

w0 = torch.tensor([[4.,5.,-83.],[6.,7.,2.],[0,0,1]])

x0 = torch.tensor([3.,2.,1.])

x1 = torch.matmul(w0,x0)
print(x1)

for i,ele in enumerate(x1):
    if(ele<=0):
        x1[i] =0


print(x1)

# print(x1[0])


w1 = torch.tensor([ [ 3.,-7.,2.   ],
                    [0,0,1]])

x2 = torch.matmul(w1,x1)
print(x2)

for i,ele in enumerate(x2):
    if(ele<=0):
        x2[i] =0

print(x2)

cc= x2[:-1]
print(cc)

c = torch.sum(cc)
c.backward()

# x2.backward()
# print(x0.grad)
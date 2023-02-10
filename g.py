import torch


x0 = torch.randn(2, 1)


w0 = torch.randn(4, 2, requires_grad=True)


x1 = torch.matmul(w0,x0)


w1 = torch.randn(1,4,requires_grad = True)

x2 = torch.matmul(w1,x1)

c = torch.sum(x2)




c.backward()
print(w0.grad)
print(w1.grad)
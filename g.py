import torch




w = torch.randn(4, 2, requires_grad=True)
x0 = torch.randn(4, 2)


x1 = w*x0

print(x1)




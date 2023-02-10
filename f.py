# numpy
import numpy as np
import torch



np.random.seed(777)

N, D = 3, 4
rb = 7

# x = np.random.randn(N, D)
# y = np.random.randn(rb, D)
# z = np.random.randn(N, D)

# a = x * y
# b = a + z
# c = np.sum(b)


x = torch.randn(N, D, requires_grad=True)
y = torch.randn(N, D)
z = torch.randn(N, D)

print(x)
print(y)


a = x * y
b = a + z
c = torch.sum(b)


print(a)


print("c",c)

c.backward()
print(x.grad)

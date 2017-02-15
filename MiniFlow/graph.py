import torch


x = torch.Tensor(5, 3)
x = torch.rand(5, 3)

print(x)

print(x.size())

y = torch.rand(5, 3)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)

import torch

def pow_reducer(x):
    return x.pow(3).sum()


inputs = torch.rand(2, 2)
print(inputs)
H = torch.autograd.functional.hessian(pow_reducer, inputs)
print(H)

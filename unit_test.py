import torch
from torch.autograd import Variable 

# useful in verifying GAN training behavior

def train_step(params):
    for name, p in params:
        change = torch.Tensor([2])
        p += change.expand(p.size())

    return params

class VariablesChangeException(Exception):
  pass

def var_change_helper(vars_change, params, device='cpu'): 
    initial_params = [(name, p.clone()) for (name, p) in params]

    params = train_step(params)

    # check if variables have changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
          if vars_change:
            assert not torch.equal(p0.to(device), p1.to(device))
          else:
            assert torch.equal(p0.to(device), p1.to(device))
        except AssertionError:
          raise VariablesChangeException("{var_name} {msg}".format(var_name=name, 
                msg='did not change!' if vars_change else 'changed!'))

if __name__=='__main__':
    params = [('test_param',torch.randn(2,3))]
    var_change_helper(False, params)
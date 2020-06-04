import torch
import torch.nn as nn
from json import loads, dumps
import numpy as np 
import h5py 
from collections import OrderedDict

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.sequential = nn.Sequential(nn.Conv2d(1, 32, 5), 
                                        nn.Conv2d(32, 64, 5), 
                                        nn.Dropout(0.3))
        self.layer1 = nn.Conv2d(64, 128, 5)
        self.layer2 = nn.Conv2d(128, 256, 5)
        self.fc = nn.Linear(256*4*4, 128)
    
    def forward(self, x):
        
        output = self.sequential(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        
        return output

model = NeuralNet()

a = model.state_dict()
a = dict(a)

def _to_numpy(a):
    for key, value in a.items():
        if isinstance(value, dict):
            _to_numpy(value)
        else:
            print(key)
            a[key] = value.cpu().detach().numpy()

def _to_tensor(a):
    for key, value in a.items():
        if isinstance(value, dict):
            _to_numpy(value)
        else:
            print(key)
            a[key] = torch.Tensor(value)

_to_numpy(a)

file = 'assets/test.h5'
save_dict_to_hdf5(a, filename=file)
# print(a)
# print(loads(dumps(a)))

dd = load_dict_from_hdf5(file)
_to_tensor(dd)
dd = OrderedDict(dd)
print(dd)
model.load_state_dict(dd)

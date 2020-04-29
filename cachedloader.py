import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import numpy as np
import ctypes

'''
Apparently PyTorch's dataloader classes performs lazy dataloading. 
This script caches the loaded data so that that items are not reprocessed in successive
iterations. 

Moreover, this script allows for the cache to be shared among multiple processes using
a shared memory object (courtesy the native multiprocessing module). So, this helps when
there are multiple workers performing dataloading.
'''

class MyDataset(Dataset):
    def __init__(self):
        shared_array_base = mp.Array(ctypes.c_float, num_data*data_dim)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())        
        shared_array = shared_array.reshape(num_data, data_dim)

        self.shared_array = torch.from_numpy(shared_array)
        self.use_cache = False

    def __getitem__(self, index):
        if not self.use_cache:
            print('Filling cache for index {}'.format(index))
            # Add loading logic here (transforms, pre-processing etc.)
            
            # no locks needed because PyTorch's workers would be operating on different indices
            self.shared_array[index] = torch.randn(data_dim) # dummy processed data
        
        x = self.shared_array[index]
        return x

    def __len__(self):
        return num_data


if __name__=='__main__':
    num_data, data_dim = 10, 3

    dataset = MyDataset()
    loader = DataLoader(
        dataset,
        num_workers=2,
        shuffle=False
    )

    for epoch in range(2):
        for idx, data in enumerate(loader):
            print('Epoch {}, idx {}, data {}'.format(epoch, idx, data))

        if epoch == 0:
            loader.dataset.use_cache = True
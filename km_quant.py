import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import random, csc_matrix, csr_matrix
from scipy import stats

'''
Each layer has its weights quantized individually
Use like following:

    for module in model.children():
        weight = module.weight.data.numpy()
        mat = apply_weight_sharing(weight, bits=5)
        module.weight.data = torch.from_numpy(mat)
'''
def apply_weight_sharing(weight, bits=5):

    shape = weight.shape
    mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
    
    min_ = min(mat.data)
    max_ = max(mat.data)
    space = np.linspace(min_, max_, num=2**bits)
    
    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
    kmeans.fit(mat.data.reshape(-1,1))
    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
    
    mat.data = new_weight
    return mat.toarray()

if __name__=='__main__':
    weight = random(30,40,density=0.5).toarray()
    print(weight)
    print(apply_weight_sharing(weight))
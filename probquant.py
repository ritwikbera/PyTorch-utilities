import numpy as np 

def quantize(weight, limits=None):

    if not limits:
        limits = [np.floor(weight), np.ceil(weight)]

    weight_min, weight_max = limits[0], limits[1]
    try:
        assert (weight_max > weight_min)
    except AssertionError:
        weight_max, weight_min = weight_min, weight_max

    p = (weight - weight_min)/(weight_max - weight_min)
    return weight_max if np.random.binomial(1, p) else weight_min

def quantize_layer(weight):
    return np.vectorize(quantize)(weight)

if __name__=='__main__':
    import torch
    import torch.nn as nn 
    weight = nn.Parameter(torch.Tensor([1.1,2.2,3.5,4.1,3.7]))
    print(quantize_layer(weight.detach().numpy()))

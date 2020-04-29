import numpy as np 

def prune_model(model, prune_percentage):
    alive_parameters = []
    # iterate over layers in a model sequentially from input to output
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        tensor = p.data.numpy()
        alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
        alive_parameters.append(alive)

    all_alives = np.concatenate(alive_parameters)
    percentile_value = np.percentile(abs(all_alives), prune_percentage)

    for name, param in model.named_parameters():
    	if 'bias' in name:
    		continue
    	prune_layer(param, threshold=percentile_value)

def prune_layer(weight, threshold):
	tensor = weight.data.numpy()
	mask = np.where(abs(tensor) < threshold, 0, np.ones(tensor.shape))
	weight.data = torch.from_numpy(tensor*mask)

if __name__=='__main__':
    import torch
    import torch.nn as nn
    weight = nn.Parameter(torch.Tensor([1,2,3,4,3]))
    prune_layer(weight, 2)
    print(weight)
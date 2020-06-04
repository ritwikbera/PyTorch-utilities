## PyTorch Tools

### Tools for Compressing Models

* [_Pruning_](prune.py) : Utility to prune low-magnitude weights in a layer.

* [_Probalistic Quantization_](probquant.py) : A demo tool to implement probabilistic quantization of weights to keep the weight statistics unbiased.

* [_K-Means Quantization_](km_quant.py) : Idea introduced in _Deep Compression_ paper to reduce number of unique weights to be stored for a NN model.

### Tools for Debugging

* [_Gradient Traceback_](gradviz.py) : Print all gradients in computation graph via recursive backtracing.

* [_Computational Graph Visualization_](visualize_graph.py) : Visualizes the PyTorch computation graph onto a PDF file. It does not require _Tensorboard_ but requires _GraphViz_ which can be installed with ```sudo apt install graphviz```.

* [_Memory Profiling with PyTorch Hooks_](checkpointing.py) : Useful in optimizing models, can help visualize memory usage during checkpointed training as well (Note: This is is not working with latest PyTorch update).

* [_Unit test to verify parameter behavior during training_](unit_test.py) : Useful while training GANs or during transfer learning where only a certain subset of parameters need to be tuned during training.

### Tools for aiding Training

* [_HDF5 Weights Import Tool_](h5_to_pth.py) : Import .h5 weight files into PyTorch models and export them back.

* [_DataLoader with Cache_](cachedloader.py) : Implements a caching mechanism in the dataloader, so that items once fetched and tranformed, from the dataset are stored in memory and are not re-processed by the dataloader. If memory limitations remain, an LRU cache (using an OrderedDict) may be used instead of a full array.

* [_Knowledge Distillation_](kdistil.py) : Template for knowledge distillation training. Used in __Parallel WaveNet__ training, among other to reduce model size. Useful only for models with softmax outputs. Anneal temperature as training progresses for stable gradients.
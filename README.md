## PyTorch Tools

### Tools for Compressing Models

* _Pruning_ : Utility to prune low-magnitude weights in a layer.

* _Probalistic Quantization_ : A demo tool to implement probabilistic quantization of weights to keep the weight statistics unbiased.

* _K-Means Quantization_ : Idea introduced in _Deep Compression_ paper to reduce number of unique weights to be stored for a NN model.

### Tools for Debugging

* _Memory Profiling with PyTorch Hooks_ : Useful in optimizing models, can help visualize memory usage during checkpointed training as well (Note: This is is not working with latest PyTorch update).

* _Unit test to verify parameter behavior during training_ : Useful while training GANs or during transfer learning where only a certain subset of parameters need to be tuned during training.

### Tools for aiding Training

* _DataLoader with Cache_ : Implements a caching mechanism in the dataloader, so that items once fetched and tranformed, from the dataset are stored in memory and are not re-processed by the dataloader. If memory limitations remain, an LRU cache (using an OrderedDict) may be used instead of a full array.

* _Knowledge Distillation_ : Template for knowledge distillation training. Used in __Parallel WaveNet__ training, among other to reduce model size. Useful only for models with softmax outputs. Anneal temperature as training progresses for stable gradients.
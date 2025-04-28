# ML Coding Exercises
Practice work for brushing up some skills.

## GMM
See `GMM.ipynb` as the entry point. 

Mock data from multiple gaussians (2D and 3D) with additive noise and implement the EM algo to fit some of varying complexity of the gaussian (e.g. start with k means, progress to a full covariance matrix). Plot the sampled points and the fit.

## Positional Encoding Exploration
See `PositionalEncoding.ipynb`

Visualize absolute sinusoidal positional encodings along with RoPE. Also visualize the dot products between learned absolute posititional encodings.

## NanoGPT Reimplementation
Reimplement the nanoGPT model from https://github.com/karpathy/nanoGPT without referencing it directly. Train a language model on Tiny Shakespeare: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt and generate text from it.

We implement the model from scratch and have a lite config for training on CPU and a heavy version for training on GPU.

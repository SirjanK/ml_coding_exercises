# ML Coding Exercises
Practice work for brushing up some skills.

## GMM
Mock data from multiple gaussians (2D and 3D) with additive noise and implement the EM algo to fit some of varying complexity of the gaussian (e.g. start with k means, progress to a full covariance matrix). Plot the sampled points and the fit.

## HMM & TIMIT Exercises
Use the TIMIT dataset to fit a custom HMM on it. Implement https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4a0e6b18b5ba5761796415c8fc3de12cb2d1efa5 from scratch, replicate results.

We also write a variant solution using DNNs to outperform this.

Finally, we take a pretrained Wav2Vec2 and fine tune it to perform the best.

## MNIST Fitting
We fit the following models on MNIST. Only libraries we're allowed to use is numpy.

1. GMM to model generative distr of pre-extracted features
2. LR
3. feed forward neural net (simple)
4. CNN (simple)
5. Visual transformer (not as suited for MNIST, but for practice)

## Stable Diffusion
We first implement from scratch (this time pytorch allowed) a diffusion model and show it works on MNIST. We then fine tune an off the shelf diffusion model on a custom image dataset.

## Fine Tuning Lanugage Model
We take an open source language model and fine tune it on a custom dataset (messenger data).

## Distributed Training Algorithms
TODO:
* literature survey
* brainstorm simple exercises to try

## RL
spinning up in RL from Open AI - implement your own versions of the algos and test it out on the different environments.

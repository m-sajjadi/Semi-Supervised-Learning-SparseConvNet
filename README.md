This is the implementation of our NIPS 2016 paper:

M. Sajjadi, M. Javanmardi, and T. Tasdizen. *Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning.* Advances in Neural Information Processing Systems. 2016.

This implementation of *mutual-exclusivity* and *transformation/stability* loss functions are based on the 'SparseConvNet':

SparseConvNet: A spatially-sparse convolutional neural network

https://github.com/btgraham/SparseConvNet/wiki  
https://github.com/btgraham/SparseConvNet/wiki/Installation  

The Installation process of our code is exactly the same as 'SparseConvNet'.

Here we have provided the code for experiments on CIFAR10, CIFAR100, MNIST, SVHN and NORB. 

For these datasets, first you need to prepare the datasets according to 'README.md' files in the 'Data' folder. Then you need to run the following commands:

```
make mnist
make cifar10
make cifar100
make svhn
make norb
```

Hyper-parameters of these experiments can be modified in 'mnist.cpp', 'cifar10.cpp', 'cifar100.cpp', 'svhn.cpp' and 'norb.cpp'. Improtant hyper-parameters can be set in the begining of each file.

- **epoch**: if set to a nonzero value, the training process starts from specified epoch.
- **cudaDevice**: which GPU to be used
- **batchSize**: size of batch

- **n_labeled**: number of samples from training set to be used as labeled set.
- **nt**: number of times each unlabeled sample should be repeated

- **lambda_mx**: lamda for mutual-exclusivity loss function
- **lambda_ts**: lamda for transformation/stability loss function

- **lab_aug**: boolean variable specifying whether to use data aumentation for labeled set or not
- **unlab_aug**: boolean variable specifying whether to use data aumentation for ulabeled set or not

If you want to look inside the code, the loss functions are defined in 'SoftmaxClassifier.cu'.

If you have any questions, contact Mehdi Sajjadi at:
mehdi@sci.utah.edu
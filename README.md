compareMvsPy, line-for-line comparison of machine learning algorithms in matlab and python.
Author: Mike Hughes, mike@michaelchughes.com 

# About
Side-by-side comparison of Expectation Maximization (EM) for training a Gaussian Mixture Model. Available for both Matlab and Python.

Allows users to compare/contrast speed, readability, and syntax for two very popular numerical software packages on a real-world machine learning task. Both implementations have the same arguments and use the same random number seeds, to allow comparison of correctness and repeatability as well.

# Requirements
Matlab

Python 2.x, with numpy and scipy

# Quick Start
From the command line, run the provided shell script

    $ ./EasyDemo.sh

This trains a 3 component full-covariance GMM in both Matlab and Python, printing progress to stdout in a standard format:  iterations done, seconds elapsed, log likelihood objective function value.

## Expected output

```
------------------------------------ Python 
EM for Mixture of 3 Gaussians | seed=8675309
    1/10 after 0 sec | -2.6887638683e+04
    2/10 after 0 sec | -2.5588164358e+04
    3/10 after 0 sec | -2.4580057922e+04
    4/10 after 0 sec | -2.4211864028e+04
    5/10 after 0 sec | -2.4140367411e+04
    6/10 after 0 sec | -2.4116683206e+04
    7/10 after 0 sec | -2.4108606045e+04
    8/10 after 0 sec | -2.4104915065e+04
    9/10 after 0 sec | -2.4102758013e+04
   10/10 after 0 sec | -2.4101339372e+04
w =  [ 0.2438  0.4181  0.338 ]
 ------------------------------------ Matlab 
EM for Mixture of 3 Gaussians | seed=8675309
     1/10 after 0 sec | -2.6887638683e+04 
     2/10 after 0 sec | -2.5588164358e+04 
     3/10 after 0 sec | -2.4580057922e+04 
     4/10 after 0 sec | -2.4211864028e+04 
     5/10 after 0 sec | -2.4140367411e+04 
     6/10 after 0 sec | -2.4116683206e+04 
     7/10 after 0 sec | -2.4108606045e+04 
     8/10 after 0 sec | -2.4104915065e+04 
     9/10 after 0 sec | -2.4102758013e+04 
    10/10 after 0 sec | -2.4101339372e+04 
w =    0.2438    0.4181    0.3380
```

Both implementations should yield *exactly* the same results, since we provide the same input and the same random seed.

# Data Input
Either script will take as input data in MAT file format. Specify the filepath as the first argument.

## Provided data
ToyData3Clusters-N6kD2.mat is a simple toy dataset of 3 well-separated clusters in 2D space. 6000 examples exist (2000 per cluster). This dataset is mostly provided as a quick way to demo the code's functionality.

ImgPatchData-N50kD64.mat is a dataset of 50,000 image patches from the Berkeley Segmentation dataset. Each row is a 64-dim vector, corresponding to a flattened version of an original 8x8 image patch with its mean removed. This dataset is provided to assess how both implementations scale to larger datasets.
#!/bin/bash
# EasyDemo.sh
#  Run EM for GMM in both Python and Matlab
#  using *exactly* the same input data and alg parameters (K=3 clusters, 10 iterations)
#  Simply a demonstration of how to use provided scripts

echo " ------------------------------------ Python "
python runEMforGMM.py ToyData3Clusters-N6kD2.mat 3 10 python-trained-model.mat 8675309

echo " ------------------------------------ Matlab "
matlab -nodesktop -nosplash -r "runEMforGMM ToyData3Clusters-N6kD2.mat 3 10 matlab-trained-model.mat 8675309; exit;"

# To restore a wonky Unix terminal prompt after running matlab from cmd line,
# you may need to uncomment the line below
# stty sane;
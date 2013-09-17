'''
Run EM algorithm for fitting Gaussian mixture model with K components

Author: Mike Hughes (mike@michaelchughes.com)

Usage
--------
Run as a script at the command line
>> python runEMforGMM.py datafilename K Niter savefilename seed

Expected Output
--------
Saves learned model parameters to a MAT file (specified by savefilename)

Arguments
--------
   datafilename  :  string path to .MAT file containing
                      observed data matrix X (N x D)
                      each row is an observed D-dim vector to cluster

   K             :  integer # of components in mix model
   Niter         :  max # of iterations to run
   savefilename  :  string path to .MAT file to save final mixture model
                       with fields .w, .mu, and .Sigma
   seed          :  integer seed for random number generation (used for init only)
'''

import argparse
import time
import random
import numpy as np
import scipy.linalg
import scipy.io

CONVERGE_THR = 1e-8
MIN_COVAR    = 1e-8
EPS          = 1e-15

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('datafilename', help='string path to .MAT file containing observed data matrix X (each row is an observed D-dim vector to cluster)')
  parser.add_argument('K', type=int, help='num of clusters')
  parser.add_argument('Niter', type=int, help='num of iterations')
  parser.add_argument('savefilename', type=str, help='path for resulting matfile')
  parser.add_argument('seed', type=int, help='random seed (for repeatable runs)')
  parser.add_argument('--MIN_COVAR', type=float, default=MIN_COVAR )
  return parser.parse_args()

def main( datafilename, K, Niter, savefilename, seed ):
  print 'EM for Mixture of %d Gaussians | seed=%d' % (K, seed);
  X = scipy.io.loadmat( datafilename )['X']
  X = X.newbyteorder('=').copy() # This makes sure X is aligned for fast linalg ops
  loglik = -np.inf*np.ones( Niter )

  Resp = init_responsibilities( X, K, seed)
  tstart = time.time()
  for t in xrange( Niter ):
    model = Mstep( X, Resp )
    Resp, loglik[t] = Estep( X, model )
    
    print '%5d/%d after %.0f sec | %.10e' % (t+1, Niter, time.time()-tstart, loglik[t])    
    deltaLogLik = loglik[t] - loglik[t-1]
    if deltaLogLik < CONVERGE_THR:
      break
    if deltaLogLik < 0:
      print 'WARNING: loglik decreased!'
  scipy.io.savemat( savefilename, model, oned_as='row' )
  np.set_printoptions(precision=4)
  print 'w = ', model['w']
  return model, loglik

def init_responsibilities( X, K, seed):
  ''' Randomly sample K observations which act as initial cluster centers
  '''
  N,D = X.shape
  random.seed( seed )
  rowIDs = random.sample( xrange(N), K ) #without replacement
  mu = X[rowIDs, : ]

  logResp = np.zeros( (N, K) )
  for k in xrange( K ):
    logResp[:,k] = loggausspdf( X, mu[k,:], np.eye(D) )
  logPrPerRow = logsumexp( logResp, axis=1 )
  Resp = np.exp( logResp - logPrPerRow[:,np.newaxis] )
  return Resp

###################################################
def Estep(X, model):
  ''' Compute responsibilities given model parameters
  
      Returns
      -------
      Resp : N x K matrix, whose rows sum to one, where
              Resp[n,k] = Pr( Z[n]=k | X[n])
      logEv : scalar log evidence of the model on this data
  '''
  w = model['w']
  mu = model['mu']
  Sigma = model['Sigma']

  N = X.shape[0]
  K = mu.shape[0]
  logResp = np.zeros( (N, K) )
  for k in xrange( K ):
    logResp[:,k] = loggausspdf( X, mu[k,:], Sigma[:,:,k] )
  logResp += np.log( w )

  logPrPerRow = logsumexp( logResp, axis=1 )
  Resp = np.exp( logResp - logPrPerRow[:,np.newaxis] )
  return Resp, np.sum(logPrPerRow)
  
def Mstep(X, Resp):
  ''' Update GMM parameters given responsibilities
  
      Returns
      -------
      dict of GMM parameters 
        w : len-K vector
        mu : KxD matrix, each row is cluster center
        Sigma : DxDxK array, Sigma[:,:,k] = covar matrix for comp k
  '''
  N,D = X.shape
  K = Resp.shape[1]

  Nk = np.sum( Resp, axis=0) + EPS
  w  = Nk/N
  mu = np.dot( Resp.T, X ) / Nk[:,np.newaxis]
  Sigma = np.zeros( (D,D,K) )
  for k in xrange( K ):
    Xdiff = X - mu[k]
    Xdiff *= np.sqrt( Resp[:,k] )[:,np.newaxis]
    Sigma[:,:,k] = dotXTX( Xdiff )/Nk[k] + MIN_COVAR*np.eye(D)
  return dict( w=w, mu=mu, Sigma=Sigma )

###################################################
def loggausspdf( X, mu, Sigma):
  ''' Calc log p( x | mu, Sigma) for each row of matrix X
      Returns
      --------
      logpdfPerRow : N x K matrix, where
        logpdfPerRow[n,k] = p( X[n] | nth data item assigned to cluster k)
  '''
  N,D = X.shape
  dist, cholSigma = distMahal( X, mu, Sigma )
  logdetSigma = 2*np.sum( np.log( np.diag(cholSigma) ) )
  logNormConst = -0.5*D*np.log(2*np.pi) - 0.5*logdetSigma
  logpdfPerRow = logNormConst - 0.5*dist
  return logpdfPerRow
  
def distMahal( X, mu, Sigma ):
  ''' Calc mahalanobis distance: (x-mu)^T Sigma^{-1} (x-mu)
       for each row of matrix X
  '''
  N,D = X.shape
  Xdiff = X - mu
  cholSigma = scipy.linalg.cholesky( Sigma, lower=True)
  Q = np.linalg.solve( cholSigma, Xdiff.T ) # 2x speedup possible over scipy.solve_tri
  Q *= Q  # better to do this in place than as sum(Q**2), since Q is *big* and latter requires a copy
  distPerRow = np.sum( Q, axis=0 )
  return distPerRow, cholSigma

def dotXTX(X):
  ''' Fast matrix multiply of X^T * X
      Args
      -------
      X : N x D matrix
      
      Returns
      -------
      R : D x D matrix, where R = X^T * X
  '''
  return scipy.linalg.fblas.dgemm( 1.0, X.T, X.T, trans_b=True)

def logsumexp( logA, axis=None):
  ''' Compute log( sum(exp(logA))) in numerically stable way
  '''
  logA = np.asarray( logA )
  logAmax = logA.max( axis=axis )

  if axis is None:
    logA = logA - logAmax
  elif axis==1:
    logA = logA - logAmax[:,np.newaxis]
  elif axis==0:
    logA = logA - logAmax[np.newaxis,:]
  assert np.allclose( logA.max(), 0.0 )
  logA = np.log( np.sum( np.exp( logA ), axis=axis )  )
  return logA + logAmax

if __name__ == '__main__':
  ''' Entry-point when script run as executable from cmd line
  '''
  args = parse_args()
  MIN_COVAR = args.MIN_COVAR
  main( args.datafilename, args.K, args.Niter, args.savefilename, args.seed )

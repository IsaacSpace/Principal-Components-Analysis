# -*- coding: utf-8 -*-
"""pca_.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Rof1tk0on6xhNzKbW95QYuo5e3zNtFJp
"""

import numpy 
from numpy.linalg import eig 


class PCA:
  """Principal Component Analysis
    This is a standard Principal Component Analysis implementation
    Parameters
    ----------
    n_components : int (optional)
        Number of components to keep. If not specified, all components are kept
    Attributes
    ----------
    eigvalues : array
        eigenvalues of covariance matrix ordered
    eigvectors : array
        eigenvectors of covariance matrix ordered by eigenvalues
    weights : array
        weights for mapping
    data : array
        original dataset
  """
  def __init__(self, n_components):
    assert isinstance(n_components, int), ValueError("n_components must be integer.")
    self.data = None 
    self.weights = None
    self.eigenvalues = None
    self.eigenvectors = None 
    self.__components = n_components

  def __CalcEigValues(self, covariance_matrix):
    self.eigenvalues, self.eigenvectors = eig(covariance_matrix)
    self.eigenvalues = numpy.abs(self.eigenvalues)
    indices = self.eigenvalues.argsort()[::-1]   
    self.eigenvalues = self.eigenvalues[indices]
    self.eigenvectors = self.eigenvectors[:,indices]

  def __CalcWeights(self):
    self.weights = self.eigenvectors[0:self.__components, :]

  def fit(self, dataset):
    self.data = dataset 
    cov_matrix = numpy.cov(dataset, rowvar=False)
    self.__CalcEigValues(cov_matrix)
    self.__CalcWeights()

  def transform(self, dataset):
    if(self.data is None):
      print("Fit is not done.")
      exit(0)
    transformed_data = numpy.matmul(dataset, self.weights.T)
    return transformed_data
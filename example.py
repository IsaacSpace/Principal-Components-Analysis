import numpy 
from pca import PCA 


# we'll create a random dataset of 5 variables and 100 samples
random_dataset = numpy.random.rand(100, 5)

# define a pca object and specify a number of components
pca_ = PCA(n_components = 2)

# fit the model using dataset
pca_.fit(dataset = random_dataset)

# transform the dataset
new_dataset = pca_.transform(dataset = random_dataset)

# print the new and old data shapes
print("Original shape:{}, new shape: {}".format(random_dataset.shape, new_dataset.shape))
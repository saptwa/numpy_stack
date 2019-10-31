import numpy as np
from datetime import  datetime

######### numpy basics #########

# python list
L = [1, 2, 3]

# equivalent numpy array
A = np.array([1, 2, 3])

# loop through the list
for e in L:
    print("element of list L: {}".format(e))

# loop through the numpy array
for e in A:
    print("element of numpy array A: {}".format(e))

# adding an element to the list (doesn't work for numpy arrays)
L.append(4)
print("list L after appending 4: {}".format(L))

# another way to append to the list is adding another list to it
# (doesn't work for numpy arrays)
L = L + [5]
print("list L after concatenating [5] to it: {}".format(L))

# adding each element of the list to itself
L2 = []
for e in L:
    L2.append(e + e)

print("L2 (each element of L added to itself): {}".format(L2))

# adding each element of the numpy array to itself
print("A + A: {}".format(A + A))
# adding two lists does concatenation, adding two numpy arrays does element-wise addition

# multiplication with a scalar
# for lists it extends the list by creating 'n' copies of it and appending them to the list
print("2 * L: {}".format(2 * L))
# for numpy arrays it does element-wise multiplication with the scalar
print("2 * A: {}".format(2 * A))

# element-wise squaring
# for lists
L3 = []
for e in L:
    L3.append(e * e)
print("L3 (element-wise squaring of the elements in L3: {}".format(L3))

# for numpy arrays
print("Element-wise squaring of the elements in the numpy array, A: {}".format(A**2))

# sq-rt
print("Element-wise square root of the elements in the numpy array, A: {}".format(np.sqrt(A)))
# log
print("Element-wise log of the elements in the numpy array, A: {}".format(np.log(A)))
# exponential
print("Element-wise exponential of the elements in the numpy array, A: {}".format(np.exp(A)))

a = np.array([1, 2])
b = np.array([2, 1])

# dot product
dot = 0
for e, f in zip(a, b):
    dot += e * f
print("dot product of a: {} & b: {} => {}".format(a, b, dot))

# another way
# a * b => element-wise multiplication => [2,2]
# np.sum(a) => sums over all the elements of a
print("dot product of a: {} & b: {} => {}".format(a, b, np.sum(a * b)))
# the sum() function is an instance method of the nnumpy array itself
print("dot product of a: {} & b: {} => {}".format(a, b, (a * b).sum()))
# using the dot() method
print("dot product of a: {} & b: {} => {}".format(a, b, np.dot(a, b)))
print("dot product of a: {} & b: {} => {}".format(a, b, a.dot(b)))
print("dot product of a: {} & b: {} => {}".format(a, b, b.dot(a)))

# calculate the cosine of the angle between two vectors a & b
a_magnitude = np.sqrt((a * a).sum())
b_magnitude = np.sqrt((b * b).sum())
print("cosine of the angle between a & b: {}".format((np.dot(a, b)) / (a_magnitude * b_magnitude)))

# using the linalg.norm() function
a_magnitude = np.linalg.norm(a)
b_magnitude = np.linalg.norm(b)
print("cosine of the angle between a & b: {}".format((np.dot(a, b)) / (a_magnitude * b_magnitude)))

print("angle between a & b: {}".format(np.arccos(np.dot(a, b) / (a_magnitude * b_magnitude)))) # measured in radians

########### dot product speed comparison ###########

x = np.random.randn(100)
y = np.random.randn(100)
T = 100000

# using for loops
t0 = datetime.now()
result = 0
for t in range(T):
    for e,f in zip(x, y):
        result += e * f
dt1 = datetime.now() - t0

# using np.dot()
t0 = datetime.now()
for t in range(T):
    np.dot(x, y)
dt2 = datetime.now() - t0

print("Time taken using for loop: {}".format(dt1.total_seconds()))
print("Time taken using for np.dot(): {}".format(dt2.total_seconds()))

print("dt1 / dt2: {}".format(dt1 / dt2))

########### matrices ###########

M = np.array([[1, 2], [3, 4]])
L = [[1, 2], [3, 4]]
# indexing in lists ==> L[0] = [1, 2], L[0][0] = 1
# indexing in numpy matrix ==> M[0][0] = 1, M[0,0] = 1

# numpy has a matrix data type as well
M2 = np.matrix([[1, 2], [3, 4]])
# the numpy documentation advises against using np.matrix()
# if you find a numpy matrix convert it to a numpy array
A = np.array(M2)
print("A: {}, \n transpose of A: {}".format(A, A.T))

########### different ways of creating numpy arrays ###########

# create a numpy array of length 10 containing all 0s
Z_array = np.zeros(10)
# create a numpy 2D array of shape (10,10) containing all 0s
Z_matrix = np.zeros((10, 10))
# create a numpy 2D array of shape (10,10) containing all 1s
Z1_matrix = np.ones((10, 10))
# create a numpy 2D array of shape (10,10) containing random numbers in the interval [0, 1]
Z_rand_matrix = np.random.random((10, 10))
# create a numpy 2D array of shape (10,10) containing random numbers
# sampled from a Gaussian distribution with mean 0 and variance 1.
# Notice how the argument is not a tuple
G_rand_matrix = np.random.randn(10, 10)
G_rand_matrix.mean()  # mean
G_rand_matrix.var()  # variance

########### matrix multiplication ###########

# C(i,j) = ∑ᵢ A(j,i) B(i,k)
# (i,j)th entry of C is the dot product of row A(j,:) and column B(:,k)
# C = np.dot(A, B) = np.inner(A, B)

# element by element multiplication
# for C(i,j) = A(i,j) * B(i,j) ==> C = A * B

# inverse
A_inv = np.linalg.inv(A)
A.dot(A_inv)  # Identity matrix

# determinant
np.linalg.det(A)

# diagonal matrix
np.diag(A)  # A = [[1, 2], [3, 4]] ==> np.diag(A) = [1,4] (numpy array containing only the diagonal elements)
np.diag([1, 4])  # [[1, 0], [0, 4]]

# outer product
np.outer(a, b)

# trace of a matrix = sum of the diagonal elements
np.diag(A).sum()
np.trace(A)

############## eigenvalues and eigenvectors ##############

Y = np.random.randn(100, 3)
# covariance
covariance = np.cov(Y.T)

# eigenvalues, eigenvectors = np.linalg.eig(C)
# or
# eigenvalues, eigenvectors = np.linalg.eigh(C)
# eigh is for symmetric and hermitian matrices
# symmetric means A = transpose of A
# hermitian means A = conjugate transpose of A

# covariance is a symmetric matrix
# the below returns a tuple ==> the 1st tuple contains 3 eigenvalues,
# the 2nd tuple contains the eigenvectors stored in the columns
np.linalg.eigh(covariance)
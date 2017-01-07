import theano.tensor as T
from theano import *
##For scalar
x = T.dscalar('x')
y = 1/(1+T.exp(-x))
dy = T.grad(y,x)
grad_softmax = function([x],dy)
print grad_softmax(10)

#For Matrix
x_matrix = T.dmatrix('x_matrix')
y_sum = T.sum(1/(1+T.exp(-x_matrix)))
dy_matrix = T.grad(y_sum,x_matrix)
grad_softmax_matrix = function([x_matrix],dy_matrix)
print grad_softmax_matrix([[1,2],[3,4]])

import theano.tensor as T
from theano import *
d = T.dmatrix('a')
arg_max = d.argmax(axis=1)
arg_max_f = function([d],arg_max)
print arg_max_f([[1,2,3],[4,5,6],[9,3,2],[10,15,4]])

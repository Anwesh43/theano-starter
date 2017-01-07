import theano.tensor as T
from theano import *
x = T.dscalar('x')
y = 1/(1+T.exp(-x))
dy = T.grad(y,x)
grad_softmax = function([x],dy)
print grad_softmax(10)

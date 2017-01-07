import theano.tensor as T
from theano import *
x = T.dscalar('x')
y = x**3+x**2
dy = T.grad(y,x)
gradF = function([x],dy)
print gradF(2)

import theano.tensor as T
x = T.dmatrix('x')
f = 1/(1+T.exp(-x))
print f.eval({x:[[1,0],[2,3]]})

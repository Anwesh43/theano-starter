from theano import function
import theano.tensor as T
a = T.dmatrix('a')
b = T.dmatrix('b')
add = a+b
print add.eval({a:[[1,0],[2,3]],b:[[2,0],[5,6]]})

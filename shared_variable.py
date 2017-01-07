from theano import *
import theano.tensor as T
s = shared(0)
print s.get_value()
inc = T.iscalar('inc')
incrementor = function([inc],s,updates=[(s,inc+s)])
decrementor = function([inc],s,updates=[(s,s-inc)])
print incrementor(5)
print incrementor(10)
print decrementor(6)
print s.get_value()
b = T.scalar(dtype=s.dtype)
a = T.scalar(dtype=s.dtype)
state_func = s**3 + b**2
stateF = function([a,b],state_func,givens=[(s,a)])
print stateF(3,4)
print s.get_value()

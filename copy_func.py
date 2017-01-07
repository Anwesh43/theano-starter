from theano import *
import theano.tensor as T
s = shared(1)
mul = T.scalar(dtype=s.dtype)
multiplier = function([mul],s,updates=[(s,s*mul)],on_unused_input='ignore')
print multiplier(2)
print multiplier(6)
print s.get_value()
ns = shared(100)
multiplier_copy = multiplier.copy(swap={s:ns})
print multiplier_copy(6)
print ns.get_value()
print s.get_value()

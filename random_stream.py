import theano.tensor as T
from theano import *
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=1000)
r_uv = srng.uniform((2,2))
r_nd = srng.normal((2,2))
r_uniform = function([],r_uv)
r_normal = function([],r_nd)
print r_uniform()
print r_normal()
print r_normal()

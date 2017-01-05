import theano.tensor as T
from theano import function
x = T.dmatrix('x')
y = T.dmatrix('y')
diff = x-y
squareDiff = x**2-y**2
cubeDiff = x**3 - y**3
f = function([x,y],[diff,squareDiff,cubeDiff])
print f([[1,2],[3,4]],[[0,1],[2,3]])

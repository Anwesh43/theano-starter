import theano.tensor as T
from theano import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
k = 0
output_dict = {}
dataset = load_iris()
X = np.array(dataset.data)
Y  = np.array(dataset.target)

for i in range(0,len(dataset.target)):
    if not(output_dict.has_key(dataset.target[i])):
        output_dict[dataset.target[i]] = 0
    output_dict[dataset.target[i]] = output_dict[dataset.target[i]]+1
k = len(output_dict)
ind_y = np.zeros((len(Y),k))
for i in range(0,len(Y)):
    ind_y[i,Y[i]] = 1
X_train,X_test,Y_train,Y_test = train_test_split(X,ind_y)
base = 4
pred = None
w = None
b = None
pickle_file = 'lr_theano.pickle'
try:
    with open(pickle_file) as f:
        lrObj = pickle.load(f)
        pred = lrObj['pred']
        w = lrObj['w']
        b = lrObj['b']
except:
    x = T.dmatrix('x')
    y = T.dmatrix('y')

    w = theano.shared(np.random.randn(base,k))
    b = theano.shared(0.0)
    o = x.dot(w)+b
    p_1 = 1/(1+T.exp(-o))
    cost = -(y*(T.log(p_1))+(1-y)*T.log(1-p_1)).mean()+(w**2).sum()
    prediction = p_1.argmax(axis=1,keepdims=True)
    gw,gb = T.grad(cost,[w,b])
    train = function(inputs=[x,y],outputs=[cost,prediction],updates=[(w,w-0.1*gw),(b,b-0.1*gb)])
    print 'training output'
    for i in range(10000):
        trainOutput = train(X_train,Y_train)
        print trainOutput[0]


    pred = function([x],prediction)
    f = open(pickle_file,'w')
    pickle.dump({'prediction':prediction,'w':w,'b':b,'pred':pred},f)
pred_y = pred(X_test)
correct = 0
print pred_y
print len(pred_y)
print len(Y_test)
for i in range(0,len(pred_y)):
    index = pred_y[i]
    index = index[0]
    if(Y_test[i][index] == 1):
        correct+=1
accuracy = (correct*1.0)/len(Y_test)
print 'accuracy is {0}'.format(accuracy)

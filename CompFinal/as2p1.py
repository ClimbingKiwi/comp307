from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

X = [] 
y = [] 
Xt = []
yt = []

irisTraining = open("iris-training.txt", "r")
with irisTraining as filehandle:  
    for line in filehandle:
       
        current_line = line[:-1]
        line_array = current_line.split('  ')
        X.append([float(i) for i in line_array[:-1]])
        y.append(line_array[-1]) 
                
irisTraining.close()

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                   hidden_layer_sizes=(16),max_iter=100,learning_rate='adaptive',momentum=0.9)

clf.fit(X, y)                         

irisTest = open("iris-test.txt", "r")
with irisTest as filehandle:  
    for line in filehandle:
       
        current_line = line[:-1]
        line_array = current_line.split('  ')
        Xt.append([float(i) for i in line_array[:-1]])
        yt.append(line_array[-1]) 
                
irisTest.close()

print(clf.predict(Xt)) 

print(clf.score(Xt,y))

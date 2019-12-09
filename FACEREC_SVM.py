import dlib
import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

vect_files = [x for x in os.listdir("vectors/")]
vect_files = sorted(vect_files)
outputs=[]
count=1
repres=[]
for y in vect_files:
	with open("vectors/"+y,"rb") as filehandle:
		repre = pickle.load(filehandle)
		repres.extend(repre[:10])
		outputs.extend([count]*len(repre[:10]))
		count+=1
print("loaded all the vectors.")
accs=[]
from sklearn.metrics import accuracy_score
from progress.bar import Bar
oplen = len(outputs)
X_train = []
Y_train = []
X_test = []
Y_test = []
print(oplen)
x=0
# Using 7 images to train, testing on 3
while x<oplen:
	X_train.extend(repres[x:x+7])
	Y_train.extend(outputs[x:x+7])
	X_test.extend(repres[x+7:x+10])
	Y_test.extend(outputs[x+7:x+10])
	x+=10

# for t_size in [0.55,0.6,0.7,0.8,0.9]:
# 	X_train, X_test, y_train, y_test = train_test_split(repres, outputs, test_size=t_size)

# print("X_train",len(X_train),"Y_train",len(Y_train))
# print("X_test",len(X_test),"Y_test",len(Y_test))

nsamples, nx, ny = np.shape(X_train)
new_X_train = np.reshape(X_train,(nsamples,nx*ny))

print(np.shape(new_X_train))
print(np.shape(Y_train))
clf = SVC(gamma='scale', decision_function_shape='ovo') 
clf.fit(new_X_train, Y_train)
print(np.shape(X_test))

nsamples, nx, ny = np.shape(X_test)
new_X_test = np.reshape(X_test,(nsamples,nx*ny))

predictions = clf.predict(new_X_test)
print(predictions)
print(Y_test)
acc = accuracy_score(Y_test,predictions)
print("accuracy",acc)

pickle.dump(clf,open('newSVMdump.sav','wb'))
	# print("----")
# print("----")
# 	bar.next()
# bar.finish()
# avgacc = sum(accs)/len(accs)
# print(t_size,avgacc)
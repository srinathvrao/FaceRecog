import dlib
import numpy as np
import cv2
import os
import pickle
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

vect_files = [x for x in os.listdir("../output/")]
# vect_files = ["0.pkl","1.pkl","2.pkl"]
vect_files = sorted(vect_files)
outputs=[]
count=0
repres=[]
for y in vect_files:
	with open("../output/"+y,"rb") as filehandle:
		repre = pickle.load(filehandle)
		repres.extend(repre)
		outputs.extend([count]*len(repre))
		count+=1
print("loaded all the vectors.")
accs=[]
from sklearn.metrics import accuracy_score
from progress.bar import Bar
oplen = len(outputs)
x=0

# for t_size in [0.55,0.6,0.7,0.8,0.9]:
X_train, X_test, y_train, y_test = train_test_split(repres, outputs,stratify=outputs, test_size=0.2)
print(len(X_train),len(y_train),len(X_test),len(y_test))
clf = SVC(gamma='scale', decision_function_shape='ovo',probability=True) 
print("fitting svm")
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
# print(predictions)
# print(Y_test)
acc = accuracy_score(y_test,predictions)
print("accuracy",acc)

dump(clf,'all_clf.sav')
	# print("----")
# print("----")
# 	bar.next()
# bar.finish()
# avgacc = sum(accs)/len(accs)
# print(t_size,avgacc)
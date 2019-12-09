from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.utils import to_categorical
import pickle
import numpy as np
import os

train_in=[]
train_op=[]
onlyfiles = [x for x in os.listdir("vectors/")]
onlyfiles = sorted(onlyfiles)

classes = [i for i in os.listdir("facenet/input/")]

num_faces = len(classes)

model = Sequential([
	Dense(512, activation='relu'),
  	Dense(256, activation='relu'),
  	Dense(128, activation='relu'),
  	Dense(num_faces, activation='softmax'),
	])

model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
	)

c=0
for y in onlyfiles:
	with open("vectors/"+y,"rb") as filehandle:
		t_in = pickle.load(filehandle)
		train_in += t_in
		print(len(t_in),len(t_in[0]),len(train_in))
		train_op += [c]*len(t_in)
	c+=1

print("train_in len",len(train_in),len(train_in[0]))
print("train_op len",len(train_op),)

model.fit(
	np.array(train_in),
	to_categorical(np.array(train_op)),
	epochs=5,
	batch_size=16
	)

model.save("facerec_51.h5")

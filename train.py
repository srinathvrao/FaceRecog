from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.utils import to_categorical
import pickle
import numpy as np

model = Sequential([
	Dense(128, activation='relu'),
  	Dense(128, activation='relu'),
  	Dense(128, activation='relu'),
  	Dense(8, activation='softmax'),
	])

model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
	)

train_op=[]
train_in=[]
with open("facenet/input_vects/train_vects","rb") as filehandle:
	train_in = pickle.load(filehandle)
with open("facenet/input_vects/train_ops","rb") as filehandle:
	train_op = pickle.load(filehandle)
print("train_in len",len(train_in))
print("train_op len",len(train_in))
model.fit(
	np.array(train_in),
	to_categorical(np.array(train_op)),
	epochs=5,
	batch_size=32
	)


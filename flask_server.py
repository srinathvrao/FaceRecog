from flask import Flask, render_template, request, jsonify
import cv2
import json
import base64
from PIL import Image
from io import StringIO
import dlib
import datetime
import pickle
from sklearn.svm import SVC
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score,make_scorer
from sklearn.model_selection import GridSearchCV
from flask_pymongo import PyMongo
import datetime
from datetime import timedelta  

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://srinath:srinath@localhost:27017/myDatabase"
mongo = PyMongo(app)
import os
import io
from PIL import Image
from array import array


face_detector = dlib.get_frontal_face_detector()
pose_predictor_68_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def whirldata_face_detectors(img, number_of_times_to_upsample=1):
	return face_detector(img, number_of_times_to_upsample)
def whirldata_face_encodings(face_image,num_jitters=1):
	face_locations = whirldata_face_detectors(face_image)
	print("face locations",face_locations)
	pose_predictor = pose_predictor_68_point
	predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
	try:
		return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors]
	except Exception as e:
		return []


def readimage(f):
	return bytearray(f)

@app.route("/image",methods=['POST'])
def sendResult():
	# print("got an image")
	# print("HELLOOOOO \n\n")
	# print(request)
	dic = request.data
	bytes = readimage(dic)
	image = Image.open(io.BytesIO(bytes))
	image.save("test.png")
	img = cv2.imread("test.png")
	clf = SVC(kernel='rbf',C=1e15,gamma=10)
	# params = {'C' : [1e5,1e6,1e7,1e8,1e9,1e10,1e10,1e12,1e14,1e16,1e18,1e20], 'gamma' : [1e-3,1e-1,1,10,100,1e3,1e5,1e7,1e9,1e11] }
	# grid = GridSearchCV(estimator = clf,param_grid = params, scoring = make_scorer(accuracy_score))
	with open('my_dumped_classifier.pkl', 'rb') as fid:
		grid = pickle.load(fid)
		clf_best = grid.best_estimator_
		repre = whirldata_face_encodings(img)
		# print(len(repre),type(repre))
		if len(repre)!=0:
			# for re in repre:
			test_op = clf_best.predict(np.array(repre))
			for label in test_op:
				print(label)
			# for label in test_op:

		else:
			print("no face")
		
	return "Hello world2"

@app.route("/")
def displ():
	return "Hello world"

if __name__ == "__main__":
	print(0,"sainath")
	print(1,"srinath")
	print(2,"midha")
	app.run("192.168.1.20",port=8083)

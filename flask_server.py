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
from os import listdir
from os.path import isfile, join
from datetime import timedelta
import copy
from keras.models import load_model

from gevent.pywsgi import WSGIServer


model = load_model('facerec.h5')

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/Attendance"
mongo = PyMongo(app)
import os
import io
from PIL import Image
from array import array
import base64


face_detector = dlib.get_frontal_face_detector()
pose_predictor_68_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def whirldata_face_detectors(img, number_of_times_to_upsample=1):
	return face_detector(img, number_of_times_to_upsample)
def whirldata_face_encodings(face_image,num_jitters=1):
	face_locations = whirldata_face_detectors(face_image)
	pose_predictor = pose_predictor_68_point
	predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
	try:
		return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors]
	except Exception as e:
		return []


def readimage(f):
	return bytearray(f)

def calculateAttendance(label, in_time, out_time):
	print('\nINSIDE CALCULATE ATTENDANCE\n')
	in_dt = datetime.datetime.strptime(in_time,"%H:%M")
	out_dt = datetime.datetime.strptime(out_time,"%H:%M")
	print(in_dt,out_dt)
	if(abs(calTimeDelta(in_dt,out_dt)) <= 30):
		return
	maps = mongo.db.mapping.find({"all":"all"})
	for doc in maps:
		print(doc[label])
		data = mongo.db.timetable.distinct("3_"+doc[label])
		period_dt = []
		for period in data:
			for i in range(1,9):
				dt = datetime.datetime.strptime(period[str(i)],"%H:%M")
				period_dt.append(dt)
		in_period = 0
		out_period = 0
		for i in range(0,8):
			if in_dt > period_dt[i]:
				if abs(calTimeDelta(in_dt,period_dt[i]))<=10:
					in_period = i+1
				else:
					in_period = i+2
			#elif abs(calTimeDelta(out_dt,period_dt[i]))<10:
			if out_dt < period_dt[i]:
				if abs(calTimeDelta(out_dt,period_dt[i]))<=10:
					out_period = i+1
				elif abs(calTimeDelta(out_dt,period_dt[i-1]))<=10:
					out_period = i-1
				else:
					out_period = i
				break


		now = datetime.datetime.now()
		dt_str  = now.strftime("%d-%m-%Y")
		print(dt_str,"to be updated..", {"class":"3_"+doc[label]})
		docs = mongo.db.date.find({"class":"3_"+doc[label]})
		docx_up = {}
		for docx in docs:
			docx_up = copy.deepcopy(docx)
			print(in_period,out_period)
			for i in range(in_period, out_period+1):
				in_array = copy.deepcopy(docx["3_"+doc[label]][dt_str][str(i)])
				in_array.append(label)
				docx_up["3_"+doc[label]][dt_str][str(i)] = in_array
			myquery = {"3_"+doc[label]:docx["3_"+doc[label]]}
			newvalues = { "$set": {"3_"+doc[label]: docx_up["3_"+doc[label]]} }
			print(str(label),"recognized entering, marked present at",dt_str)
			mongo.db.date.update_one(myquery,newvalues)
		return "hi"
	# datedb =  mongo.db.date.distinct(label of class+"/"+date)
	# for i in range(1,4):
	# 	if i>=in_period and i<=out_period:
	# 		datedb[i].append(label of student)
	#
	#
	# myquery = {"3_cse_c":  { "22-08-2019" }}
	# output = mongo.db.find(myquery)
	# print(output)
	# newvalues = { "$set": {"cse_c_"+str(label) : { "in": dt_str , "present": "True" }} }
	# print(str(label),"recognized entering, marked present at",dt_str)
	# mongo.db.attendance.update_one(myquery,newvalues)

def calTimeDelta(now_time, out_time):
	td = now_time - out_time
	mins = (td.total_seconds()//60)
	return mins

@app.route("/imagesend", methods=['GET','POST'])
def sendcalc():
	calculateAttendance("2","09::58","11::35")

@app.route("/image", methods=['POST'])
def sendResult():
	# print("got an image")
	# print("HELLOOOOO \n\n")
	# print(request)
	# dic = request.data
	# bytes = readimage(dic)
	# image = Image.open(io.BytesIO(bytes))
	##dic = request.data
	dic = request.json
	print(type(dic))
	print()
	#arr = np.array(dic['arr'])
	#picnp = np.fromstring(dic.getvalue(), dtype=np.uint8)
	#bytes = readimage(dic)
	#bytes = io.BytesIO(dic)
	print()
	print(dic['time'])
	print()
	picnp = np.fromstring(base64.b64decode(dic['img']), dtype=np.uint8)
	#image = Image.open(io.BytesIO(dic))
	#print(type(bytes))
	#print(type(picnp))
	#print()
	#image = Image.open(bytes)
	#image.save("test.png")
	#img = cv2.imread("test.png")
	img = cv2.imdecode(picnp, 1)
	#cv2.imshow('image',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	clf = SVC(kernel='rbf',C=1e15,gamma=10)
	# onlyfiles = [f for f in listdir("pics") if isfile(join("pics/", f))]
	# le = len(onlyfiles) + 1
	# image.save("pics/test%d.png" % le)
	#2img = cv2.imread("2.jpg")
	# params = {'C' : [1e5,1e6,1e7,1e8,1e9,1e10,1e10,1e12,1e14,1e16,1e18,1e20], 'gamma' : [1e-3,1e-1,1,10,100,1e3,1e5,1e7,1e9,1e11] }
	# grid = GridSearchCV(estimator = clf,param_grid = params, scoring = make_scorer(accuracy_score))
	#camera1 = False
	with open('my_dumped_classifier2.pkl', 'rb') as fid:
		grid = pickle.load(fid)
		clf_best = grid.best_estimator_
		repre = whirldata_face_encodings(img)
		if len(repre)!=0:
			#test_op = clf_best.predict(np.array(repre))
			predictions = model.predict(np.array(repre)).tolist()
			print(predictions)
			test_op = []
			for i in range(len(predictions)):
				if max(predictions[i]) > 0.6:
					test_op.append(predictions[i].index(max(predictions[i])))
				else:
					print('UREGISTERED PERSON\n')

			print(test_op)
			if dic['camera'] == 1:#camera1: # camera 1 triggered (coming in)
				print('CAMERA 1')
				for label in test_op:
					docs = mongo.db.attendance.distinct("cse_c_"+str(label))
					for doc in docs:
						if doc['present']=="False":
							now = datetime.datetime.now()
							dt_str  = now.strftime("%H:%M")
							#myquery = {"cse_c_"+str(label):  { "in": "" , "present": "False" }}
							print('initial in time',doc['in'],'label',label)
							myquery = {"cse_c_"+str(label):  { "in": doc['in'] , "present": "False","out":doc["out"] }}
							newvalues = { "$set": {"cse_c_"+str(label) : { "in": dt_str , "present": "True" , "out":doc["out"]}} }
							print(str(label),"recognized entering, marked present at",dt_str)
							mongo.db.attendance.update_one(myquery,newvalues)
						elif doc['present'] == "True" and calTimeDelta(datetime.datetime.now() ,datetime.datetime.strptime(doc['in'],"%H:%M")) > 2:
							print('penalize')
							## TODO: Penalize
						#else:
						#	print("ALREADY MARKED")
			elif dic['camera'] == 2: # camera 2 triggered (going out)
				print('CAMERA 2')
				for label in test_op:
					docs = mongo.db.attendance.distinct("cse_c_"+str(label))
					for doc in docs:
						if doc['present']=="True": #and calTimeDelta(datetime.datetime.now() ,datetime.datetime.strptime(doc['out'],"%H:%M")) == False:
							myquery = {"cse_c_"+str(label) : doc}
							doc2 = {}
							doc2['in'] = doc['in']
							doc2['out'] = datetime.datetime.now().strftime("%H:%M")
							doc2['present'] = "False"
							newvalues = { "$set": {"cse_c_"+str(label) : doc2} }
							print(str(label),"recognized leaving, marked absent")
							calculateAttendance(str(label),doc['in'],datetime.datetime.now().strftime("%H:%M"))
							mongo.db.attendance.update_one(myquery,newvalues)
						elif doc['present'] == "False" and calTimeDelta(datetime.datetime.now() ,datetime.datetime.strptime(doc['out'],"%H:%M"))>2:
						 	print('penalize')
							## TODO: Penalize
			else:
				print('invalid camera')


		else:
			print("no face")

	return "Hello world2"

@app.route("/")
def displ():
	return "Hello world"

if __name__ == "__main__":
	print(0,"sainath")
	print(1,"midha")
	print(2,"srinath")
	#app.run("192.168.1.6",port=8083)
	http_server = WSGIServer(('192.168.43.7', 8083), app)
	http_server.serve_forever()

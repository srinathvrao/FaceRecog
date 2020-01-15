from flask import Flask, render_template, request, jsonify
import cv2
import json
import base64
from PIL import Image
from io import StringIO
import datetime
import pickle
from joblib import load
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
import warnings
warnings.filterwarnings("ignore")

from gevent.pywsgi import WSGIServer
import insightface

analysis_model = insightface.app.FaceAnalysis()
analysis_model.prepare(ctx_id=-1,nms=0.4)

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/Attendance"
mongo = PyMongo(app)
import os
import io
from PIL import Image
from array import array
import base64

imgc = 0

def readimage(f):
	return bytearray(f)

def calculateAttendance(person):
	print('\nINSIDE CALCULATE ATTENDANCE\n')
	in_dt = datetime.datetime.strptime(person['in'],"%H:%M")
	out_dt = datetime.datetime.strptime(person['out'],"%H:%M")
	print('\nin_time:',in_dt,'\nout_time:',out_dt,'\n')
	if(abs(calTimeDelta(in_dt,out_dt)) <= 30):
		print('difference less than 30')
		return
	print('difference greater than 30')

	time_table = mongo.db.timetable.distinct(person['cid'])
	for i,times in enumerate(time_table):
		time_table[i] = datetime.datetime.strptime(times,'%H:%M')
	
	in_period = 0
	out_period = 0

	for i in range(0,8):
		if in_dt >= time_table[i]:
					if abs(calTimeDelta(in_dt,time_table[i]))<=30:
						in_period = i
					else:
						in_period = i+1
		if out_dt < time_table[i]:
			if abs(calTimeDelta(out_dt,time_table[i]))<=10:
				out_period = i
			else:
				out_period = i-1
			break
	
	print('in_period,out_period',in_period,out_period)
	now = datetime.datetime.now()
	dt_str  = now.strftime("%d-%m-%Y")

	print(dt_str,"to be updated..", {person['rno']})
	date_list = mongo.db.date.find_one({'cid':person['cid']})
	try:
		attendance_list = date_list[dt_str]
		for i in range(in_period,out_period + 1):
			attendance_list[i].append(person['rno'])
	except:
		attendance_list = [[],[],[],[],[],[],[],[]]
		for i in range(in_period,out_period + 1):
			attendance_list[i].append(person['rno'])
	mongo.db.date.update_one({'_id':date_list['_id']},{'$set':{dt_str:attendance_list}})
	return "hi"

def calTimeDelta(now_time, out_time):
	td =  out_time - now_time
	mins = (td.total_seconds()//60)
	return mins

@app.route("/imagesend", methods=['GET','POST'])
def sendcalc(): #used for testing
	person = {}
	calculateAttendance(person) 

@app.route("/image", methods=['POST'])
def sendResult():
	global imgc
	dic = request.data
	print()
	picnp = np.fromstring(base64.b64decode(dic), dtype=np.uint8)
	img = cv2.imdecode(picnp, 1)
	cv2.imwrite("saved/" + "input_"+ str(imgc)+ ".png",img)
	faces = analysis_model.get(img)
	model = load('012_clf.sav')
	print("[INFO] Number of faces : ", len(faces))
	for idx, face in enumerate(faces):
		boxx = (face.bbox.astype(np.int).flatten())
		x1 = boxx[0]
		y1 = boxx[1]
		x2 = boxx[2]
		y2 = boxx[3]
		img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
		cv2.imwrite("saved/test"+str(idx)+"_"+str(imgc)+".jpg",img)
		repre = face.embedding
		if len(repre)!=0:
			predictions = model.predict_proba([repre])[0]
			print("[INFO] Predictions: ",predictions)
			print("[INFO] Predicted ID: ",predictions.argmax())
			if camera1: #TODO: check if csmera 1
				print('\nCamera 1\n')
				now_time = datetime.datetime.now().strftime('%H:%M')
				person = mongo.db.attendance.find_one({'id':predictions.argmax()})
				if person['present'] == False:
					print('\nIdentified',person['cid'],' marking in time at time')
					person['in'] = now_time
					mongo.db.attendance.update_one({'_id':person['_id']},{'$set':{'in':person['in'],'present':True}})
				else:
					if calTimeDelta(person['in'],now_time) > 5:
						print('\nPENALIZE:',person['cid'])
						#TODO: Penalize

			elif camera2: #TODO: check if camera 2
				print('\nCamera 2\n')
				now_time = datetime.datetime.now().strftime('%H:%M')
				person = mongo.db.attendance.find_one({'id':predictions.argmax()})
				if person['present']:
					print('\nIdentified',person['cid'],' marking out time at time')
					person['out'] = now_time
					mongo.db.attendance.update_one({'_id':person[id]},{'$set':{'out':person['out'],'present':False}})
					calculateAttendance(person)
				else:
					print('\nAlready marked out')
					#TODO: do something when person is already marked out

	print("\n\n")
	print("imagecount:",imgc)
	imgc+=1
	return "Hello world2"

@app.route("/")
def displ():
	return "Hello world"

if __name__ == "__main__":
	print(0,"sainath")
	print(1,"midha")
	print(2,"srinath")
	http_server = WSGIServer(('192.168.43.68', 8083), app)
	http_server.serve_forever()

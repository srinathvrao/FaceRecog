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

def calTimeDelta(now_time, out_time):
	td = now_time - out_time
	mins = (td.total_seconds()//60)%60
	return mins
imgc = 0
@app.route("/imagesend", methods=['GET','POST'])
def sendcalc():
	calculateAttendance("2","09::58","11::35")

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
	print("\n\n\n")
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

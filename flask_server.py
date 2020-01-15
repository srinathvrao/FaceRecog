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
#from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

from gevent.pywsgi import WSGIServer
# import insightface

# analysis_model = insightface.app.FaceAnalysis()
# analysis_model.prepare(ctx_id=-1,nms=0.4)
# embed_model = insightface.model_zoo.get_model('arcface_r100_v1')
# embed_model.prepare(ctx_id = -1)	
# detect_model = insightface.model_zoo.get_model('retinaface_r50_v1')
# detect_model.prepare(ctx_id = -1, nms=0.4)


# model = load_model('facerec_51.h5')

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
		print('difference less than 30')
		return
	print('difference greater than 30')
	maps = mongo.db.mapping.find()
	print('before maps')
	print('maps:',maps)
	for doc in maps:
		print('in maps')
		if label in doc:
			print(doc[label])
			data = mongo.db.timetable.distinct("3_"+doc[label])
			print('data',data)
			period_dt = []
			for period in data:
				print('period',period)
				dt = datetime.datetime.strptime(period,"%H:%M")
				period_dt.append(dt)
				# for i in range(1,9):
				# 	print(i,data[i])
				# 	dt = datetime.datetime.strptime(data[i],"%H:%M")
				# 	period_dt.append(dt)
			in_period = 0
			out_period = 0
			print('period_dt',period_dt)
			for i in range(0,8):
				if in_dt >= period_dt[i]:
					if abs(calTimeDelta(in_dt,period_dt[i]))<=30:
						in_period = i
					else:
						in_period = i+1
				#elif abs(calTimeDelta(out_dt,period_dt[i]))<10:
				if out_dt < period_dt[i]:
					if abs(calTimeDelta(out_dt,period_dt[i]))<=10:
						out_period = i
					# elif abs(calTimeDelta(out_dt,period_dt[i-1]))<=10:
					# 	out_period = i-1
					else:
						out_period = i-1
					break

			print('in_period,out_period',in_period,out_period)
			now = datetime.datetime.now()
			dt_str  = now.strftime("%d-%m-%Y")
			print(dt_str,"to be updated..", {"class":"3_"+doc[label]})
			docs = mongo.db.date.find()
			docx_up = {}
			for docx in docs:
				docx_up = copy.deepcopy(docx)
				print(in_period,out_period)
				for i in range(in_period, out_period + 1):
					in_array = copy.deepcopy(docx["3_"+doc[label]][dt_str][i])
					in_array.append(label)
					docx_up["3_"+doc[label]][dt_str][i] = in_array
				myquery = {"3_"+doc[label]:docx["3_"+doc[label]]}
				newvalues = { "$set": {"3_"+doc[label]: docx_up["3_"+doc[label]]} }
				print(str(label),"recognized entering, marked present at",dt_str)
				mongo.db.date.update_one(myquery,newvalues)
			return "hi"

	print('for ended')
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
	print('now time',now_time)
	td = out_time - now_time
	print(td)
	print('seconds:',td.total_seconds())
	mins = (td.total_seconds()//60)
	print('difference in mins:',mins)
	return mins
imgc = 0
@app.route("/imagesend", methods=['GET','POST'])
def sendcalc():
	person = mongo.db.attendance.find_one({'pid':'1'})
	print(person)

	now = datetime.datetime.now()
	# dt_str  = now.strftime("%d-%m-%Y")
	# person['in'] = datetime.datetime.now().strftime('%H:%M')
	time_table = mongo.db.timetable.distinct(person['cid'])
	
	# print(dt_str,"to be updated..", {"class":"3_"+doc[label]})
	date_list = mongo.db.date.find_one({'cid':person['cid']})
	in_period = 2
	out_period = 5
	dt_str = '16-01-2020'
	try:
		attendance_list = date_list[dt_str]
		for i in range(in_period,out_period + 1):
			attendance_list[i].append(person['rno'])
	except:
		attendance_list = [[],[],[],[],[],[],[],[]]
		for i in range(in_period,out_period + 1):
			attendance_list[i].append(person['rno'])
	mongo.db.date.update_one({'_id':date_list['_id']},{'$set':{dt_str:attendance_list}})
	
	


	# print(type(time_table))
	# print(time_table)
	# period_dt = []
	# for period in time_table:
	# 	print('period',period)
	# 	dt = datetime.datetime.strptime(period,"%H:%M")
	# 	period_dt.append(dt)
	# print('pd:',period_dt)

	# for i,times in enumerate(time_table):
	# 	time_table[i] = datetime.datetime.strptime(times,'%H:%M')
	# print(time_table)

	# docs = mongo.db['mapping']
	# for doc in docs.find():
	#  	print(doc['132'])
	# calculateAttendance("132","08:30","15:00")
	#time.sleep(10)
	print('\n\nRequest handled')
	return 'test'

@app.route("/image", methods=['POST'])
def sendResult():
	global imgc
	dic = request.data
	print()
	picnp = np.fromstring(base64.b64decode(dic), dtype=np.uint8)
	img = cv2.imdecode(picnp, 1)
	# print(img.shape)
	cv2.imwrite("saved/" + "input_"+ str(imgc)+ ".png",img)
	# img = cv2.imread("saved/test.png")

	faces = analysis_model.get(img)
	# model = pickle.load(open('012_clf.sav','rb'))
	model = load('012_clf.sav')
	print("[INFO] Number of faces : ", len(faces))
	# model = grid
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
			#test_op = clf_best.predict(np.array(repre))
			predictions = model.predict_proba([repre])[0]
			print("[INFO] Predictions: ",predictions)
			print("[INFO] Predicted ID: ",predictions.argmax())
			'''
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
							myquery = {"cse_c_"+str(label):  { "in": doc['in'] , "out":doc["out"], "present": "False" }}
							newvalues = { "$set": {"cse_c_"+str(label) : { "in": dt_str ,  "out":doc["out"], "present": "True"}} }
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
						print(calTimeDelta(datetime.datetime.now() ,datetime.datetime.strptime(doc['out'],"%H:%M")))
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
			'''
	#cv2.imshow('image',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	# onlyfiles = [f for f in listdir("pics") if isfile(join("pics/", f))]
	# le = len(onlyfiles) + 1
	# image.save("pics/test%d.png" % le)
	#2img = cv2.imread("2.jpg")
	# params = {'C' : [1e5,1e6,1e7,1e8,1e9,1e10,1e10,1e12,1e14,1e16,1e18,1e20], 'gamma' : [1e-3,1e-1,1,10,100,1e3,1e5,1e7,1e9,1e11] }
	# grid = GridSearchCV(estimator = clf,param_grid = params, scoring = make_scorer(accuracy_score))
	#camera1 = False
	# with open('my_dumped_classifier2.pkl', 'rb') as fid:
	
	else:
		print("no face")
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
	#app.run("192.168.1.6",port=8083)
	http_server = WSGIServer(('localhost', 8083), app)
	http_server.serve_forever()

import dlib
import cv2
from sklearn.svm import SVC
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score,make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
import threading
import os

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
		return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors][0]
	except Exception as e:
		return []

labels = [0,1,2]
train_vects = []
train_op=[]
for label in labels:
	facesrec=0
	file_dir = "facenet/input/"+str(label)+"/"
	onlyfiles = [x for x in os.listdir(file_dir)]
	for file in onlyfiles:
		filepath = file_dir + file
		img = cv2.imread(filepath)
		repre = whirldata_face_encodings(img)
		if len(repre)!=0:
			train_vects.append(repre)
			facesrec+=1
			print(facesrec)
	train_op += [label]*facesrec


clf = SVC(kernel='rbf',C=1e15,gamma=10)
params = {'C' : [1e5,1e6,1e7,1e8,1e9,1e10,1e10,1e12,1e14,1e16,1e18,1e20], 'gamma' : [1e-3,1e-1,1,10,100,1e3,1e5,1e7,1e9,1e11] }

grid = GridSearchCV(estimator = clf,param_grid = params, scoring = make_scorer(accuracy_score))
grid.fit(np.array(train_vects),np.array(train_op))
# print(grid.cv_results_)
with open('my_dumped_classifier2.pkl', 'wb') as fid:
	pickle.dump(grid,fid)
#clf.fit(np.array(train_in),np.array(train_op))
import cv2
# cap = cv2.VideoCapture("tcp://192.168.1.18:3333")
l=[]
count=0

# def run_detector(frame):
# 	detections = whirldata_face_detectors(frame)
# 	for detection in detections:
# 		print(detection)
# 		break
		# 	x = (detection.left()) # x
		# 	y = (detection.top()) # y
		# 	w = (detection.right() - x) # width
		# 	h = (detection.bottom() - y) # height
		# 	# single_op = clf.predict(test_np)
		# 	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
		# 	repre = whirldata_face_encodings(frame,detections)
		# 	if len(repre)!=0:
		# 		test_op = clf_best.predict(np.array([repre]))
		# 		print(test_op)
# while(True):
	# count+=1
	# if count==5:
	# 	t = threading.Thread(target = run_detector, args=(frame,))
	# 	t.start()
	# 	count=0
	# Our operations on the frame come here
	# print(whirldata_face_detectors(gray))


		# break

	# Display the resulting frame
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break
# with open('sainath.txt','wb') as file:
# 	pickle.dump(l,file)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

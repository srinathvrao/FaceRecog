import dlib
import cv2
from sklearn.svm import SVC
import dlib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
face_detector = dlib.get_frontal_face_detector()
pose_predictor_68_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def whirldata_face_detectors(img, number_of_times_to_upsample=1):
	return face_detector(img, number_of_times_to_upsample)
def whirldata_face_encodings(face_image,face_locations,num_jitters=1):
	# face_locations = whirldata_face_detectors(face_image)
	pose_predictor = pose_predictor_68_point
	predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
	try:
		return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors][0]
	except Exception as e:
		return []

sainath=[]
with open("sainath.txt","rb") as file:
	sainath = pickle.load(file)
srinath=[]
with open("srinath.txt","rb") as file:
	srinath = pickle.load(file)
midha=[]
with open("midha.txt","rb") as file:
	midha = pickle.load(file)
train_op = [0]*10 + [1]*10 + [2]*10
train_in = sainath + srinath + midha

clf = SVC(kernel='rbf',C=1e15,gamma=10)
clf.fit(np.array(train_in),np.array(train_op))
import cv2
cap = cv2.VideoCapture(0)
l=[]
while(True):
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)

	# Our operations on the frame come here
	# print(whirldata_face_detectors(gray))
	detections = whirldata_face_detectors(frame)
	for detection in detections:
		x = (detection.left()) # x
		y = (detection.top()) # y
		w = (detection.right() - x) # width 
		h = (detection.bottom() - y) # height
		# single_op = clf.predict(test_np)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
		repre = whirldata_face_encodings(frame,detections)
		if len(repre)!=0:
			test_op = clf.predict(np.array([repre]))
			print(test_op)

		# break

	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
# with open('sainath.txt','wb') as file:
# 	pickle.dump(l,file)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
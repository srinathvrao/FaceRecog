import os
import dlib
import cv2
import pickle
import numpy as np

onlyfiles = [int(x) for x in os.listdir("images/")]
onlyfiles = sorted(onlyfiles)

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
sizes=[]
c=0
for x in onlyfiles:
	c=0
	vects=[]
	imagefiles = [s for s in os.listdir("images/"+str(x)+"/")]
	for y in imagefiles:
		imgpath = "images/"+str(x)+"/"+y
		img = cv2.imread(imgpath)
		repre = whirldata_face_encodings(img)
		if len(repre)!=0:
			print(imgpath)
			c+=1
			vects.append(repre[0])
	with open("vectors/vect"+str(x)+".data","wb") as file:
		sizes.append(c)
		print(len(vects),len(vects[0]))
		pickle.dump(vects,file)

print(sizes)

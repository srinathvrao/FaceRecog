import cv2
import os
import numpy as np
import faceRecognition as fr


## TRAINING

#faces,faceID=fr.labels_for_training_data('C:/Users/Srinath/Desktop/FaceRecog/trainingImages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.write('trainingData.yml')
#================================= END OF TRAINING
# Uncomment the above lines when training a face.


## TESTING

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')
name={0:"Chris Evans",1:"Robert Downey Jr.", 2: "Tom Hollands"}



#This module takes images  stored in diskand performs face recognition


## TESTING

directory = "C:/Users/Srinath/Desktop/FaceRecog/TestImages"

for path,subdirnames,filenames in os.walk(directory):
	for filename in filenames:
		if filename.startswith("."):
			print("Skipping system file")
			continue
		id=os.path.basename(path)#fetching subdirectory names
		img_path=os.path.join(path,filename)
		print("img_path: ",img_path+"\n")
		#print("id:",id)
		test_img=cv2.imread(img_path)
		if test_img is None:
			print("Image Not Loaded Properly!")
			continue
		faces_detected,gray_img=fr.faceDetection(test_img)
		for face in faces_detected:
			(x,y,w,h)=face
			roi_gray=gray_img[y:y+h,x:x+h]
			face_recognizer.read('trainingData.yml')
			label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
			#print("confidence:",confidence)
			fr.draw_rect(test_img,face)
			predicted_name=name[label]
			if(confidence>37):#If confidence greater than 37 then don't print predicted face text on screen
				print("\n==================")
				continue
			#fr.put_text(test_img,predicted_name,x,y)
			#resized_img=cv2.resize(test_img,(600,400))
			# not displaying... just printing if recognized or not.
			print("recognized " + predicted_name +" in "+img_path+"\n==================")

## END OF TESTING
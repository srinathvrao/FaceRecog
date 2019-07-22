import cv2
import numpy as np
import os
import time
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor,desiredFaceWidth=200)

FACE_DIR = "input/"

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def main():
    create_folder(FACE_DIR)
    while True:
        name = input("Enter name: ")
        face_id = input("Enter face ID for face : ")
        try:
            face_id = int(face_id)
            FACE_FOLDER = FACE_DIR + str(face_id) + "/"
            create_folder(FACE_FOLDER)
            break
        except:
            print("Invalid Input!")
            continue


    while True:
        initial_image_number = input("Starting image number: ")
        try:
            initial_image_number = int(initial_image_number)
            break
        except:
            print("Starting image number should be integer!")
            continue

    image_number = initial_image_number
    cap = cv2.VideoCapture(0)
    total_images = 10
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces)==1:
            face = faces[0]
            (x,y,w,h) = face_utils.rect_to_bb(face)
            face_image = gray[y-50:y+h+100, x-50:x+w+100]
            face_aligned = face_aligner.align(frame,gray,face)

            face_image = face_aligned
            image_path = FACE_FOLDER + name + str(image_number) + ".jpg"
            cv2.imwrite(image_path, face_image)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 3)
            cv2.imshow("Aligned Face",face_image)
            image_number += 1

        cv2.imshow("Saving",frame)
        cv2.waitKey(1)
        if image_number == initial_image_number + total_images:
            break

    cap.release()


main()

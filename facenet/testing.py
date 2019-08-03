import numpy as np
import dlib
import cv2
import glob
import scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *

detector = dlib.get_frontal_face_detector()
model = load_model('trained_weights.h5')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

threshold = 0.25

def initialize():
    database = {}

    # load all the images of students to mark into the database
    for file in glob.glob("input/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = fr_utils.img_path_to_encoding(file, model)
    return database

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def extract_face_info(img, img_rgb, database,ear):
    faces = detector(img_rgb)
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            image = img[y:y + h, x:x + w]
            name, min_dist = recognize_face(image, database)
            if ear > thresh:
                if min_dist < 0.1:
                    cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                else:
                    cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            else:
                cv2.putText(img, 'Eyes Closed', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


def recognize_face(face_descriptor, database):
    encoding = img_to_encoding(face_descriptor, model)
    min_dist = 100
    identity = None

    for name, db_enc in database.items():

        distance = np.linalg.norm(db_enc - encoding)

        print("Distance for %s is %s :" %(name, distance))

        if dist < min_dist:
            min_dist = dist
            identity = name

        if int(identity) == 1:
            return str("Rohit"), min_dist
        else
            return str('Dont know'), min_dist

def recognize():
    database = initialize()
    cap = cv2.VideoCapture(0)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    while True:
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)
        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
            extract_face_info(img, img_rgb, database,ear)
        cv2.imshow('Recognizing faces', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


recognize()

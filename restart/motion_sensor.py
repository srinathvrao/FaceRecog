import RPi.GPIO as GPIO
import time
import io
from picamera import PiCamera
import picamera.array
import numpy as np
import queue

import threading
import requests
import json
import base64
from flask import jsonify
import base64

from gpiozero import MotionSensor
from PIL import Image
import datetime

class Captures:
    def __init__(self,bytearr,timestamp):
        self.bytearr = bytearr
        self.timestamp = timestamp

#stream = io.BytesIO()
url = 'http://192.168.137.1:8083/image'

def send_pics():
    print("THREAD STARTED")
    #send pic to server
    # data = {}
    # with open('rick.png', mode='rb') as file:
    #     img = file.read()
    # data['img'] = base64.encodebytes(img).decode("utf-8")
    #
    # print(json.dumps(data))
    while True:
        #print('LOOP RUNNING')
        if Q.empty() == False:
            print('INSIDE IF')
            #print(stream)
            print('sending pics..\n')
            obj = Q.get()
            #data = obj.bytearr
            json = {'img' : base64.b64encode(obj.bytearr), 'time' : obj.timestamp, 'camera' : 1}
            #data = {"eventType": "AAS_PORTAL_START", "data": {"uid": "hfe3hf45huf33545", "aid": "1", "vid": "1"}}
            #params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103', 'format': 'xml', 'platformId': 1}
            #data = {"image":('rick.png',open('rick.png','rb'))}
            params = {'sessionKey':'9129u192849128'}

            r = requests.post(url, json = json)
            print(r)
            print('pics sent')

Q = queue.Queue()
camera = PiCamera()

#stream = picamera.array.PiRGBArray(camera)
#camera.start_preview()
pir = MotionSensor(4)

t = threading.Thread(target = send_pics,args=())
t.start()

def motion():
    while(True):
        start_time = time.time()
        print('motion')
        for i in range(1):
            stream = io.BytesIO()


            camera.capture(stream,format = 'jpeg', use_video_port=True)
            Qobj = Captures(stream.getvalue(),datetime.datetime.now())
            Q.put(Qobj)
            print(type(stream.getvalue()))
            print('Q empty:',Q.empty())

        print('pics captured')
        print(time.time() - start_time)



while True:
    while pir.motion_detected:
        print('\n\nMotion Detected\n\n')
        motion()
        if pir.motion_detected == False:
            print('\n\nNo Motion\n\n')
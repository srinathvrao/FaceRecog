import RPi.GPIO as GPIO
import time
import io
from picamera import PiCamera
import picamera.array
import numpy as np
import queue
'''GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.IN)         #Read output from PIR motion sensor
GPIO.setup(3, GPIO.OUT)         #LED output pin
print('works')
while True:
	i=GPIO.input(11)
	if i==0:                 #When output from motion sensor is LOW
		print("No intruders")
	else:
                print("Intruders") '''
import threading
import requests
import json
import base64

class Captures:
    def __init__(self,bytearr,timestamp):
        self.bytearr = bytearr
        self.timestamp = timestamp

stream = io.BytesIO()
url = 'http://192.168.1.6:8083/image'

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
            
            print(stream)
            print('sending pics..')
            #for i in range(15):
            #data = open('test.h264','rb')
            #data = open('csec%02d.jpg' % i,'rb')
            obj = Q.get()
            data = obj.bytearr
            #data = {'img' : obj.bytearr, 'time' : obj.timestamp}
            #data = {"eventType": "AAS_PORTAL_START", "data": {"uid": "hfe3hf45huf33545", "aid": "1", "vid": "1"}}
            #params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103', 'format': 'xml', 'platformId': 1}
            #data = {"image":('rick.png',open('rick.png','rb'))}
            params = {'sessionKey':'9129u192849128'}

            #print(data)

            r = requests.post(url, params=params, data = data)
            print(r)
            print('pics sent')


from gpiozero import MotionSensor
from PIL import Image
Q = queue.Queue()
camera = PiCamera()

#stream = picamera.array.PiRGBArray(camera)
#camera.start_preview()
pir = MotionSensor(4)

t = threading.Thread(target = send_pics,args=())
t.start()

def motion():
    start_time = time.time()
    print('motion')
    #for i in range(15):
    #camera.capture('test.png')
    '''camera.start_recording('test.h264')
    time.sleep(5)
    camera.stop_recording()'''
    #camera.capture_sequence(['csec%02d.jpg' % i for i in range(15)])
    #camera.stop_preview()


    camera.capture(stream,format = 'jpeg', use_video_port=True)
    Qobj = Captures(stream.getvalue(),time.time())
    Q.put(Qobj)
    print(type(stream.getvalue()))
    if Q.empty() == False:
        print('WOOHOO')
    print(Q.empty())
    #camera.capture('sigh.jpg')
    
    print('pics captured')
    print(time.time() - start_time)
    #pir.wait_for_no_motion()
    


while True:
    while pir.motion_detected:
        motion()
        pir.wait_for_no_motion()
    #pir.when_motion = motion
    #pir.wait_for_motion()
    #motion()
    
#pir.wait_for_no_motion()
#camera.close()





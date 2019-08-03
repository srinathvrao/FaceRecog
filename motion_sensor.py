import RPi.GPIO as GPIO
import time
from picamera import PiCamera
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
from gpiozero import MotionSensor
camera = PiCamera()
camera.start_preview()
pir = MotionSensor(4)
#while True:
start_time = time.time()
pir.wait_for_motion()
print('motion')
#camera.capture('test.png')
'''camera.start_recording('test.h264')
time.sleep(5)
camera.stop_recording()'''
camera.capture_sequence(['csec%02d.jpg' % i for i in range(50)])
camera.stop_preview()
#pir.wait_for_no_motion()
camera.close()


import requests
import json
import base64

#send pic to server
url = 'http://192.168.1.20:8083/image'
# data = {}
# with open('rick.png', mode='rb') as file:
#     img = file.read()
# data['img'] = base64.encodebytes(img).decode("utf-8")
#
# print(json.dumps(data))
data = open('test.h264','rb')
#data = {"eventType": "AAS_PORTAL_START", "data": {"uid": "hfe3hf45huf33545", "aid": "1", "vid": "1"}}
#params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103', 'format': 'xml', 'platformId': 1}
#data = {"image":('rick.png',open('rick.png','rb'))}
params = {'sessionKey':'9129u192849128'}

print(data)

r = requests.post(url, params=params, data=data)
print(r)

print(time.time() - start_time)

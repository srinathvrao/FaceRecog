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
pir = MotionSensor(4)
#while True:
start_time = time.time()
pir.wait_for_motion()
print('motion');
camera.capture('test.png')
pir.wait_for_no_motion()
camera.close()
print('Motion');
print(time.time() - start_time)

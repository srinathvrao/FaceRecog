from picamera import PiCamera
import time

camera = PiCamera()

camera.start_recording('capture.h264')
time.sleep(45)
camera.stop_recording()

from picamera import PiCamera
import time

camera = PiCamera()

camera.start_recording('capture_fake.h264')
time.sleep(45)
camera.stop_recording()

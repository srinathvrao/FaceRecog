from flask import Flask, render_template, request, jsonify
import cv2
import json
import base64
from PIL import Image
from io import StringIO
app = Flask(__name__)

import os
import io
from PIL import Image
from array import array

def readimage(f):
	return bytearray(f)

@app.route("/image",methods=['POST'])
def sendResult():
	# print("got an image")
	# print("HELLOOOOO \n\n")
	# print(request)
	dic = request.data
	bytes = readimage(dic)
	image = Image.open(io.BytesIO(bytes))
	image.save("test.png")
	
	return "Hello world2"

@app.route("/")
def displ():
	return "Hello world"

if __name__ == "__main__":
    app.run("192.168.1.20",port=8080)

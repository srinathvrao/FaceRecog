import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

images = os.listdir('imgs')
print(images)

for image in images:
    try:
        img = Image.open(image)
        img.load()
        im2 = img.point(lambda p: p * 2.5)
        im2.show()
    except:
        print("Cant open image!")
        exit()

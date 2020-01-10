import albumentations as A
import numpy as np
import cv2
import os, gc, time, random


dataset_dir = "input/"
images_dirs = os.listdir(dataset_dir)


def augment(aug, image):
    augmented = aug(image=image, mask=None, bboxes=None,category_id=None)
    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)
    return image_aug


random.seed(23)
medium = A.Compose([
    A.HorizontalFlip(p=1),
    A.Blur(blur_limit=35, p=1),
    A.RandomBrightness(p=1),
    A.OpticalDistortion(p=1),
    A.HueSaturationValue(p=1),
    A.GaussNoise(p=1),
    A.GaussianBlur(p=1),
    A.ImageCompression(p=1),
], p=1)

for image_dir in images_dirs:
    for img in image_dir:
        image = cv2.imread(dataset_dir + '/' + images_dir + "/"  + img)
        img_aug = augment(aug, image)
        image_name = img.split(".jpg")[0]
        cv2.imwrite(dataset_dir + '/' + images_dir + "/"  + image_name + "_augment.jpg")

print("[INFO] Done...")        

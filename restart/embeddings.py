import insightface
import cv2
import numpy as np
import os, gc, re
from tqdm import tqdm
import time
import pickle

dataset_dir = "../input2/"
images_dirs = os.listdir(dataset_dir)
# images_dirs = [str(x) for x in range(3)]
vectors_dir = "../output/"

if not os.path.exists(vectors_dir):
    os.makedirs(vectors_dir)
# Model Zoo : "https://github.com/deepinsight/insightface/wiki/Model-Zoo"
model = insightface.app.FaceAnalysis()
# model = insightface.model_zoo.get_model('arcface_r100_v1')
# set ctx_id to -1 for CPU.
ctx_id = 0

print("[INFO]Preparing Model...")
model.prepare(ctx_id=ctx_id, nms=0.3)
# model.prepare(ctx_id=ctx_id)
all_embeddings = list()

print("[INFO]Extracting embeddings: ")

for image_dir in tqdm(images_dirs):
    print("[INFO]Extracting embeddings from %s" % image_dir)
    start_time = time.time()
    embeddings = list()
    c=0
    for img in os.listdir(os.path.join(dataset_dir,image_dir)):
        # print(os.path.join(dataset_dir,image_dir,img))
        # print(os.path.join(dataset_dir,image_dir))
        # print(cv2.imread(os.path.join(dataset_dir,image_dir,img)).shape)
        # exit(0)
        c+=1
        faces = model.get(cv2.imread(os.path.join(dataset_dir,image_dir,img)))
        for idx, face in enumerate(faces):
            embeddings.append(face.embedding)
    print(c)
    pickle.dump(embeddings, open(vectors_dir+"/"+image_dir+".pkl","wb"))
    all_embeddings.append(embeddings)
    del embeddings

print("[INFO]Done...")

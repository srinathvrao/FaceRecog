import insightface
import cv2
import numpy as np
import os, gc, re
from tqdm import tqdm
import time
import pickle

dataset_dir = "input/"
# images_dirs = os.listdir(dataset_dir)
images_dirs = ["0","1","2"]
vectors_dir = "output/"

if not os.path.exists(vectors_dir):
    os.makedirs(vectors_dir)
# Model Zoo : "https://github.com/deepinsight/insightface/wiki/Model-Zoo"
model = insightface.app.FaceAnalysis()

# set ctx_id to -1 for CPU.
ctx_id = 0

print("[INFO]Preparing Model...")
model.prepare(ctx_id=ctx_id, nms=0.3)

all_embeddings = list()

print("[INFO]Extracting embeddings: ")

for image_dir in tqdm(images_dirs):
    print("[INFO]Extracting embeddings from %s" % image_dir)
    start_time = time.time()
    embeddings = list()
    for img in os.listdir(os.path.join(dataset_dir,image_dir)):
        # print(os.path.join(dataset_dir,image_dir,img))
        # exit(0)
        faces = model.get(cv2.imread(os.path.join(dataset_dir,image_dir,img)))
        for idx, face in enumerate(faces):
            embeddings.append(face.embedding)

    pickle.dump(embeddings, open(vectors_dir+"/"+image_dir+".pkl","wb"))
    all_embeddings.append(embeddings)
    del embeddings

print("[INFO]Done...")

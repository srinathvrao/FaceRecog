# Restart

## Step 1 : Dataset Collection
`collect_faces.py` runs a simple script that captures aligned faces from a live-feed.

Post `collect_faces.py` you'll have a folder `input` which contains sub folders, one for each persons face.

These sub folders can further have any number of images. 200 is the default. This can be changed.

## Step 2 : Extracting embeddings

For extracting embeddings we use `insight-face`. This is current SOTA for face recognition.

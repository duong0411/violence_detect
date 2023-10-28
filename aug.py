import cv2
from PIL import Image
import os
import random
from PIL import ImageEnhance
import tqdm
import numpy as np

def vid_loader(path):
    frames = []
    cam = cv2.VideoCapture(path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    while(cam.isOpened()):
        ret,frame = cam.read()
        if ret:
            frame = cv2.resize(frame, (512,512))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(frame)
            img = img.rotate(-90)
            frames.append(img)
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    num_frames = len(frames)
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return videodims, fourcc, fps, num_frames, frames

def writeVid(op, vid, iteration, fps, lbl, vidd):
    output_dir = "augmented/" + str(lbl) + "/" + str(vidd)  # Specify the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_path = os.path.join(output_dir, f"{op}{iteration}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videodims, num_frames = vid[0].size, len(vid[0])

    video = cv2.VideoWriter(video_path, fourcc, fps, (videodims[1], videodims[0]), 0)
    print("Creating " + f"{op}{iteration}.mp4")
    for i in range(num_frames):
        video.write(np.array(vid[iteration][i]))
    video.release()

# Modify your augmentation functions to use the updated writeVid function
def aug_rotate(n, degree, lbl, vidd):
    rotated = []
    for i in range(n):
        angle = random.uniform(-degree, degree)
        rot = [img.rotate(angle) for img in vid]
        rotated.append(rot)
        writeVid("rotate", rotated, i, fps, lbl, vidd)

def aug_brightness(n, value, lbl, vidd):
    bright = []
    for i in range(n):
        nits = random.uniform(0.5, value)
        bri = [ImageEnhance.Brightness(img).enhance(nits) for img in vid]
        bright.append(bri)
        writeVid("brightness", bright, i, fps, lbl, vidd)

# Add similar modifications to other augmentation functions

# Your main loop
datadir = "C:/Users/Admin/PycharmProjects/pythonProject5/Code/Real Life Violence Dataset"
categories = {0: "NonViolence", 1: "Violence"}
label = []

for category in tqdm(categories, desc="Augmenting videos"):
    path = os.path.join(datadir, categories[category])
    lbl = categories[category]
    label.append(lbl)
    for vids in os.listdir(path):
        vidd = vids.split('.')[0]
        videodims, fourcc, fps, num_frames, vid = vid_loader(os.path.join(path, vids))
        aug_brightness(5, 10, lbl, vidd)

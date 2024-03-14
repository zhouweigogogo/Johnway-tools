import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT_DIR = os.getcwd()

train_path = os.path.join(ROOT_DIR, "images/train")
val_path = os.path.join(ROOT_DIR, "images/val")

train = os.listdir(train_path)
val = os.listdir(val_path)

with open("train.txt", "w") as f:
    for i in train:
        img = os.path.join(train_path, i)
        f.write(img + "\n")

with open("val.txt", "w") as f:
    for i in val:
        img = os.path.join(val_path, i)
        f.write(img + "\n")

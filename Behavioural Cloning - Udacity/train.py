import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import csv
from model import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__=="__main__":
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 50
    VAL_PCT = 0.2
    ALPHA = 0.4
    LOAD_MODEL = False
    SAVE_MODEL = True
    MODEL_NAME = 'model1.h5'
    
    lines = []
    x = []
    y = []

    with open('driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines[1:]:
        center = cv2.resize(cv2.cvtColor(cv2.imread(line[0]),cv2.COLOR_BGR2RGB),(160,80))/255.0
        # left = cv2.resize(cv2.cvtColor(cv2.imread(line[1][1:]),cv2.COLOR_BGR2RGB),(160,80))/255.0
        # right = cv2.resize(cv2.cvtColor(cv2.imread(line[2][1:]),cv2.COLOR_BGR2RGB),(160,80))/255.0
        x.append(center)
        y.append(float(line[3]))
        x.append(cv2.flip(center,1))
        y.append(-1*float(line[3]))

        # x.append(left)
        # y.append(float(line[3])+ALPHA)
        # x.append(cv2.flip(left,1))
        # y.append(-1*(float(line[3])+ALPHA))

        # x.append(right)
        # y.append(float(line[3])-ALPHA)
        # x.append(cv2.flip(right,1))
        # y.append(-1*(float(line[3])-ALPHA))

    x, y = np.array(x), np.array(y)
    print(np.max(x[0]), np.min(x[0]))
    plt.imshow(x[0])
    plt.show()
    model = model()

    if LOAD_MODEL:
        print("Loading model...")
        model.load(MODEL_NAME)
        print("Model loaded.")

    train(model, x, y, lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS, val_pct=VAL_PCT, save_model=SAVE_MODEL, model_name=MODEL_NAME)

    

        
        

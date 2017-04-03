
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
get_ipython().magic('matplotlib inline')


# In[2]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint


# In[3]:

import os
import csv


# In[4]:

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader:
        samples.append(line)

from sklearn.cross_validation import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[5]:

import sklearn
import itertools
from sklearn.utils import shuffle


# In[8]:

def preprocess_image(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV reads image in BGR format
    img = img[80:80+32,0::10,:] # crop the bottom and top, subsample the width to reach 32x32x3 size
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # luminance and chrominance space
    img = img[:,:,0] # most of the information is in the luminace part
    img = cv2.equalizeHist(img) # correction of random brightness in different frames
    img = img.astype('float32')
    img = (img/255.0)-0.5  # Normalizing to mean 0 and max deviation of abs(0.5)
    img = np.array(img)
    img = img.reshape(32,32,1)
    return list(img)


# In[9]:

def generator(samples, batch_size=4):
    num_samples = len(samples)
    correction_factor = 0.3 # offset to be added for left and right cameras
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB) # OpenCV reads image in BGR format
                center_image = preprocess_image(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)  # OpenCV reads image in BGR format
                left_image = preprocess_image(left_image)
                left_angle = center_angle + correction_factor
                images.append(left_image)
                angles.append(left_angle)
                name = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)  # OpenCV reads image in BGR format
                right_image = preprocess_image(right_image)
                right_angle = center_angle - correction_factor
                images.append(right_image)
                angles.append(right_angle)   
                images.append(np.fliplr(center_image)) # Data augmentation to balance out left and right turns
                angles.append(-center_angle)   
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)   
                images.append(np.fliplr(right_image))
                angles.append(-right_angle)   
                

            # batch size of 24 images, 4 images x 3 cams x 2 perpectives(original and flipped)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)


# In[15]:

batch_size = 4 # batch size of 4 returns 24 images, 3 cams and also flipped images


# In[11]:

train_generator = generator(train_samples, batch_size=batch_size)


# In[ ]:

validation_generator = generator(validation_samples, batch_size=batch_size)


# In[12]:

model = Sequential()

#model.add(Conv2D(3, (3, 3),padding='valid', input_shape=(32, 32,1)))

model.add(Conv2D(6, (5, 5),padding='valid', input_shape=(32, 32,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('elu'))

model.add(Conv2D(16, (5, 5),padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('elu'))

model.add(Flatten())

model.add(Dense(120))
#model.add(Dropout(0.5))
model.add(Activation('elu'))


model.add(Dense(84))
#model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()


# In[16]:

steps_per_epoch = len(train_samples)/batch_size


# In[18]:

validation_steps = len(validation_samples)/batch_size


# In[13]:

checkpointer = ModelCheckpoint(filepath="./data/gen_model_{epoch:02d}.h5", verbose=1, save_best_only=False)
model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,validation_data =validation_generator,validation_steps=validation_steps, epochs=5,callbacks=[checkpointer])
#model.save('model.h5')


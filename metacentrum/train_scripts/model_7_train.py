# importing necessary libraries
# numpy for algebraic and array coputations, matplotlib for drawing figures,
# tensorflow for all machine learning functions (architecture defining, metrics),
# sklearn for function for splitting the dataset into train, test and valid,
# datetime and pytz for accessing current time

import numpy as np
from matplotlib import pyplot as plt
#import random

import tensorflow as tf

from tensorflow.keras.metrics import MeanAbsoluteError

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, TimeDistributed, BatchNormalization, Conv3D, Conv3DTranspose

from datetime import datetime
import pytz
#import os

#######################

# loadingg the data from compressed files saved on meta servers
images = np.load('imgs_256.npz')['imgs']
voxels = np.load('voxels_32.npz')['voxels']


# splitting dataset to train(80%), test(10%) and validation(10%) set
# random_state can be any positive integer, and ensures that dataset is split randomly, but always the same
X_train, X_rem, y_train, y_rem = train_test_split(images,voxels, train_size=0.8, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=42)

# deleting large variables that are no longer needed to optimize the memory consumption
del images, voxels, X_rem, y_rem

# definition of the model's architecture for singleview 2D to 3D
single2Dto3D = tf.keras.Sequential([
        layers.Input(shape = (256, 256, 3)),
        layers.Conv2D(32, kernel_size=15, padding='valid',strides=1, activation='relu'),
        layers.Conv2D(32, kernel_size=9, padding='valid',strides=1, activation='relu'),
        layers.MaxPool2D(2),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=7, padding='valid',strides=1, activation='relu'),
        layers.Conv2D(64, kernel_size=7, padding='valid', strides=1, activation='relu'),
        layers.MaxPool2D(2),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=3, padding='valid', strides=1, activation='relu'),
        layers.Conv2D(128, kernel_size=3, padding='valid', strides=1, activation='relu'),
        layers.MaxPool2D(2),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=3, padding='valid', strides=1, activation = 'relu'),
        layers.Conv2D(256, kernel_size=3, padding='valid', strides=1, activation = 'sigmoid'),
        layers.BatchNormalization(),

        layers.Reshape((20, 20, 1, 256)),

        layers.Conv3DTranspose(128, kernel_size=(3,3,8), padding='valid',strides=1, activation='relu'),
        layers.Conv3DTranspose(128, kernel_size=(3,3,9), padding='valid',strides=1, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv3DTranspose(64, kernel_size=(5,5,9), padding='valid',strides=1, activation='relu'),
        layers.Conv3DTranspose(64, kernel_size=(5,5,9), padding='valid',strides=1, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv3D(1, kernel_size=1, activation='sigmoid', padding='valid'),
        layers.Reshape((32, 32, 32))

])
single2Dto3D.compile(optimizer='adam', loss=losses.BinaryCrossentropy(), metrics='binary_crossentropy')
# Adam optimizer is used for model and BCE as both loss function and metric
comp = 'BCE'

# however it's better to use already trained model, the right paht needs to be ensured:
model_2Dto3D = tf.keras.models.load_model('2405061705.keras')


# function for predicting and merging the voxels from singleview with arithmetic mean
def merging_voxels(images):
  models = []
  merged_models = []
  for i in range(images.shape[0]):
    model = model_2Dto3D.predict(images[i])
    model = np.where(model >0.5, 1, 0).astype('uint8')
    models.append(model)
    merged_models.append(np.mean(model, axis=0))
  X_models = np.array(models)
  X_merged = np.array(merged_models)

  return X_merged, X_models

merged_train, to_merge_train = merging_voxels(X_train)
merged_valid, to_merge_valid = merging_voxels(X_valid)
merged_test, to_merge_test = merging_voxels(X_test)


model_fusion = tf.keras.Sequential([
    layers.Input(shape = (32, 32, 32)),
    layers.Reshape((32, 32, 32, 1)),
    Conv3D(16, kernel_size=4, padding='valid',strides=1, activation='relu'),
    Conv3D(32, kernel_size=4, padding='valid', strides=1, activation='relu'),
    Conv3D(64, kernel_size=4, padding='valid', strides=1, activation='relu'),

    Conv3DTranspose(64, kernel_size=4, padding='valid',strides=1, activation='relu'),
    Conv3DTranspose(32, kernel_size=4, padding='valid', strides=1, activation='relu'),
    Conv3DTranspose(1, kernel_size=4, padding='valid', strides=1, activation='sigmoid'),
    layers.Reshape((32, 32, 32))
])
# Adam optimizer is used for model and BCE as both loss function and metric
model_fusion.compile(optimizer='adam', loss=losses.BinaryCrossentropy(), metrics='binary_crossentropy')



# fitting the model using the train data and validation data, saving all info about training into history variable
history = model_fusion.fit(merged_train, y_train,
                batch_size = 32,
                epochs=200,
                shuffle=True,
                validation_data=(merged_valid, y_valid))

# Current time for file names
CET = pytz.timezone('Europe/Prague')
now = datetime.now(CET)
date_time = now.strftime("%y%m%d%H%M")

# saving the model's architecture, weights and biases
model_fusion.save('output/'+date_time+'.keras')

# plotting the train and validation loss through training from the history
fig, ax = plt.subplots()
plt.plot(history.history['loss'][5:], label='loss')
plt.plot(history.history['val_loss'][5:], label='val_loss')
ax.set_xlabel('iterations')
ax.set_ylabel('loss')
ax.legend()
ax.set_title(date_time)
plt.savefig('output/train_loss.pdf')
plt.clf()

# plotting error through the training from histroy (it's actually the same as the loss)
if comp == 'MAE':
  fig, ax = plt.subplots()
  plt.plot(history.history['mean_absolute_error'][5:], label='MAE')
  plt.plot(history.history['val_mean_absolute_error'][5:], label='val_MAE')
  ax.set_xlabel('iterations')
  ax.set_ylabel('error')
  ax.legend()
  ax.set_title(date_time)
  plt.savefig('output/train_error.pdf')
  plt.clf()
elif comp == 'BCE':
  fig, ax = plt.subplots()
  plt.plot(history.history['binary_crossentropy'][5:], label='BCE')
  plt.plot(history.history['val_binary_crossentropy'][5:], label='val_BCE')
  ax.set_xlabel('iterations')
  ax.set_ylabel('error')
  ax.legend()
  ax.set_title(date_time)
  plt.savefig('output/train_error.pdf')
  plt.clf()


# definition of metrics that will be used to evaluate the model
def MAE(X, y):
  mae = tf.keras.metrics.MeanAbsoluteError()
  mae.update_state([X], [y])
  mae.result().numpy()
  return mae.result().numpy()

def STD(X, y):
  difference = np.subtract(X, y)
  std = np.round(np.std(difference),10)
  return std

def BCE(X, y):
  bce = tf.keras.metrics.BinaryCrossentropy()
  bce.update_state([X], [y])
  bce.result().numpy()
  print('BCE = ', bce.result().numpy())
  return bce.result().numpy()

# path to the folder created by run.sh and the text file init to save our results
file_path = 'output/results.txt'

with open(file_path, 'a') as file:
      file.write(f"Model: {date_time}\n\n")
      #file.close()

# this function predicts the voxels from images with given threshold and writes the results in the results.txt and
# creates the figures of visual results and saves them as vector graphic
def treshold(restriction):
  test_pred = model_fusion.predict(merged_test)
  test_pred_tresholded = np.where(test_pred >0.5, 1, 0)
  del test_pred
  test_pred_tresholded = test_pred_tresholded.astype('uint8')

  maes.append(MAE(test_pred_tresholded, y_test))
  stds.append(STD(test_pred_tresholded, y_test))
  bces.append(BCE(test_pred_tresholded, y_test))

  with open(file_path, 'a') as file:
      file.write(f"Treshold = {restriction}\n")
      file.write(f"MAE = {MAE(test_pred_tresholded, y_test)}\n")
      file.write(f"STD = {STD(test_pred_tresholded, y_test)}\n")
      file.write(f"BCE = {BCE(test_pred_tresholded, y_test)}\n\n")
      #file.close()


      # Change this variable to any positive integer to look through the test set
      k = 0

      n = 6
      fig =  plt.figure(figsize=(10, 10))
      for i in range(n):
          ax = fig.add_subplot(4, 3, i + 1)
          plt.imshow(X_test[k][i])

      ax = fig.add_subplot(4, 3, 2*3 + 1, projection='3d')
      ax.voxels(test_pred_tresholded[k,:,:,:], edgecolor='k')

      ax = fig.add_subplot(4, 3, 3*3 + 1, projection='3d')
      ax.voxels(y_test[k,:,:,:], edgecolor='k')
  plt.savefig(f'output/figures_test_{str(restriction)[-1]}.pdf')

  del test_pred_tresholded


# there we are using function from above to iterate through some selected thresholds and save results
maes = []
stds = []
bces = []
restrictions = [0.1, 0.3, 0.5, 0.7, 0.9]

for restriction in restrictions:
   treshold(restriction)


# plotting the metrics from all the thresholds (but we won't do that)
#fig, ax = plt.subplots()
#plt.plot(restrictions,maes, label='MAE')
#plt.plot(restrictions,maes+stds, color='c', label='+-STD')
#plt.plot(restrictions,maes-stds, color='c')
#plt.fill_between(restrictions,maes-stds,maes+stds,alpha=0.3)
#ax.set_xlabel('treshold')
#ax.set_ylabel('error')
#ax.legend()
#ax.set_title(date_time)
#plt.savefig('output/maes_with_stds.pdf')
#plt.clf()


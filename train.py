import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


from IPython.display import display
from PIL import Image 
from PIL import ImageOps

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from keras.models import load_model


train_df = pd.read_csv("./data/train.csv")

#test_df = pd.read_csv('data/sample_submission.csv')

input_path = "./data/cropped_train_images"

root_in = '../data'
root_out = './'# root_out give the folder direciton of after-processed images, csv..

train_df["species"].replace({"kiler_whale": "killer_whale", "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)


import tensorflow as tf
import os
from os.path import exists


AUTOTUNE = tf.data.AUTOTUNE
image_count = len(train_df)
img_height=256
img_width=256
batch_size = 32


species_names=train_df["species"].unique()
species_dict={}
i=0
for sp in species_names:
  species_dict[sp]=i
  i = i+1

print(species_dict)

train_df["species_int"]= train_df["species"].apply(lambda x: species_dict[x])

def image_exists(imdbid):
    filepath = f"data/cropped_train_images/{imdbid}"
    return os.path.isfile(filepath)


train_df = train_df[train_df['image'].apply(image_exists)]


list_ds = (
    tf.data.Dataset.from_tensor_slices(
        (
            train_df['image'].values,
            train_df['species_int'].values
        )
    )
)

for f in list_ds.take(5):
  print(f[0].numpy())


list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size  
  return tf.image.resize_with_pad(img, img_height, img_width)
  
  
  
def process_image(image,label):
  # Load the raw data from the file as a string
  #print(image)
  #print(label)
  img = tf.io.read_file('data/cropped_train_images/'+image)
  img = decode_img(img)
  return img, label


train_ds = train_ds.map(process_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_image, num_parallel_calls=AUTOTUNE)

for tupla in train_ds.take(3):
  image=tupla[0]
  label=tupla[1]
  print("Image shape: ", image.shape)
  print("Label: ", label.numpy())

print(species_names)

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

#print(next(iter(train_ds)))

# %%
num_classes = len(species_names)

if os.path.isdir('happywhales_nnmodel'):
  print("Loading model...")
  model=load_model('happywhales_nnmodel')
else:
  model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
  ])

'''
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# %%

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1
)

model.save('happywhales_nnmodel')'''

#test

class_names=train_df["species"].unique()

i=0
hits=0
for index,row in train_df.iterrows():
  
  kimg = tf.io.read_file('data/cropped_train_images/'+row['image'])
  kimg = decode_img(kimg)

  img_array = tf.keras.utils.img_to_array(kimg)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  species1=class_names[np.argmax(score)]

  if species1==row['species']:
    hits = hits + 1

  print(row['image']+' (' + row['species'] + ') -> '+species1)
  i = i +1
  if i==1500:
    break

print('accuracy: ' + str(float(hits)/i))

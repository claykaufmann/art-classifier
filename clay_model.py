"""
This file is purely for training on the VACC. Fine-tuning, metrics, and plots
were all created in clay_model.ipynb
"""

# imports
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as K
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt

# get directories
main_direc = os.getcwd()
images_dir = os.path.join(main_direc, 'data/images/images')

# csv location
artist_csv_loc = os.path.join(main_direc, 'data/artists.csv')   

"""
Set hyperparams for the number of classes and image generators
"""

BATCH_SIZE = 64 

IMG_WIDTH = 299
IMG_HEIGHT = 299
NUM_ARTISTS = 10

# Collecting Needed Images
artists = pd.read_csv(artist_csv_loc)

# Creating a dataframe with the top 10 artists by number of paintings
artists_sort = artists.sort_values(by=['paintings'], ascending=False)

artists_top = artists_sort.head(NUM_ARTISTS) # need to add 1 so 10 classes are read in
print(artists_top)

# Images
artists_dir = os.listdir(images_dir) # Files are named after each artists

# Images DataFrame
artists_top_name = artists_top['name'].str.replace(' ', '_').values

images_df = pd.DataFrame()
for name in artists_top_name:
    images_df = pd.concat([images_df, pd.DataFrame(data={'Path': glob.glob('data/images/images/' + name + '/*'), 'Name': name})], ignore_index=True)

print(images_df)

train_df = images_df.sample(frac=0.8, random_state=200)
test_df = images_df.drop(train_df.index)

if K.backend.image_data_format() == 'channels_first':
    input_shape = (3, IMG_HEIGHT, IMG_WIDTH)
else:
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

"""
Build generators
"""

train_generator = ImageDataGenerator(rescale=1.0 / 255,
                                    rotation_range=20,
                                    zoom_range=0.05,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    horizontal_flip=True,
                                    fill_mode="nearest",
                                    validation_split=0.15,
                                    preprocessing_function=preprocess_input
                                    )

test_generator = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)

train_gen = train_generator.flow_from_dataframe(
    train_df,
    shuffle=True,
    x_col='Path',
    y_col='Name',
    class_mode='categorical',
    subset="training",
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    seed=42
)

valid_gen = train_generator.flow_from_dataframe(
    train_df,
    subset="validation",
    shuffle=True,
    x_col='Path',
    y_col='Name',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    seed=42
)

test_gen = test_generator.flow_from_dataframe(
    test_df,
    x_col='Path',
    batch_size=1,
    shuffle=False,
    class_mode=None,
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)

# Set the amount of steps for training, validation, and testing data
# based on the batch size
steps_train = train_gen.n//train_gen.batch_size
steps_valid = valid_gen.n//valid_gen.batch_size
steps_test = test_gen.n//test_gen.batch_size

"""
Hyperparameters here:
"""
N_EPOCHS = 50
LEARNING_RATE = 0.001 # 0.001 is the default for Adam set by TensorFlow
OPTIMIZER = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FUNCTION = tf.losses.CategoricalCrossentropy(from_logits=False)

# set the input for VGG
inp = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))

# load model
base_model = InceptionV3(include_top=False, input_tensor=inp, pooling='max', weights='imagenet')

# set base model to not be trainable
base_model.trainable = False

final_model = tf.keras.Sequential()
final_model.add(base_model)

final_model.add(K.layers.Flatten())
final_model.add(K.layers.BatchNormalization())
final_model.add(Dense(256, activation='relu'))
final_model.add(K.layers.Dropout(0.6))
final_model.add(Dense(NUM_ARTISTS, activation='softmax'))

final_model.summary()

# compile model
final_model.compile(
  optimizer=OPTIMIZER,
  loss=LOSS_FUNCTION,
  metrics=['accuracy']
)

# create a checkpoint for the model
checkpt = ModelCheckpoint(filepath='clay_trained_model.hdf5', save_best_only=True, verbose=1)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')

# Fit the model
history = final_model.fit_generator(
    generator = train_gen,
    steps_per_epoch=steps_train,
    validation_data = valid_gen,
    validation_steps = steps_valid,
    verbose=1,
    epochs=N_EPOCHS,
    callbacks=[checkpt, early_stop]
)

"""
    The next section creates plots for data during training
"""
# Create accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('clay_model_accuracy.jpg')

# clear plot
plt.clf()
plt.cla()
plt.close()

# create loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('clay_model_loss.jpg')

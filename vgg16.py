

# Import libraries
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
import glob
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras import regularizers, optimizers
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


artists = pd.read_csv('../cs254-final-project/data/artists.csv')
print(artists.shape)
# print(artists)

# Creating a dataframe with the top 10 artists by number of paintings
artists_top = artists.head(10)
artists_top

# Images
images_dir = '../cs254-final-project/data/images/images'
artists_dir = os.listdir(images_dir) # Files are named after each artists

# Images DataFrame
artists_top_name = artists_top['name'].str.replace(' ', '_').values

images_df = pd.DataFrame()
for name in artists_top_name:
    # print(glob.glob('../cs254-final-project/data/images/images/' + name + '/*'))

    # Method 1:
    #
    # images_df = images_df.append(pd.DataFrame(data={'Path': glob.glob('../cs254-final-project/data/images/images/' + name + '/*'), 'Name': name}), ignore_index=True)

    # Method 2:
    #
    images_df = pd.concat([images_df, pd.DataFrame(data={'Path': glob.glob('../cs254-final-project/data/images/images/' + name + '/*'), 'Name': name})], ignore_index=True)

images_df

# Create Generator


BATCH_SIZE = 64

# image dimensions?
img_width, img_height = 277, 277

train_df = images_df.sample(frac=0.8, random_state=200)
test_df = images_df.drop(train_df.index)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# instantiate neural network

# Train

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
    target_size=(img_width, img_height),
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
    target_size=(img_width, img_height),
    seed=42
)

test_gen = test_generator.flow_from_dataframe(
    test_df,
    x_col='Path',
    batch_size=1,
    shuffle=False,
    class_mode=None,
    target_size=(img_width, img_height)
)


def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    #top_model = Dense(500, activation='relu')(top_model)
    #top_model = Dense(100, activation='relu')(top_model)
    #top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)

    model = Model(inputs=conv_base.input, outputs=output_layer)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
#from livelossplot.inputs.keras import PlotLossesCallback

#step sizes:
steps_train = train_gen.n//train_gen.batch_size
steps_valid = valid_gen.n//valid_gen.batch_size
steps_test = test_gen.n//test_gen.batch_size


optimizer = keras.optimizers.Adam(learning_rate=0.001)
n_classes = 9

n_epochs = 50

vgg = create_model(input_shape,n_classes, optimizer, fine_tune=0)

#loss_plot
#v1_loss_plot = PlotLossesCallback()

#model checkpoint

v1_checkpoint = ModelCheckpoint(filepath='v1_best_weights.hdf5',
                               save_best_only = True,
                               verbose = 1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

vgg.summary()

model_history = vgg.fit_generator(
        generator = train_gen,
        steps_per_epoch = steps_train,
        validation_data = valid_gen,
        validation_steps = steps_valid,
        callbacks=[v1_checkpoint,early_stop],
        verbose=1,
        epochs = n_epochs
)

# Generate predictions
vgg.load_weights('v1_best_weights.hdf5') # initialize the best trained weights

true_classes = test_gen.classes
class_indices = train_gen.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

vgg_preds = vgg.predict(test_gen)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)

from sklearn.metrics import accuracy_score

vgg_acc = accuracy_score(true_classes, vgg_pred_classes)
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc * 100))

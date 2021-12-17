



# Import libraries
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as K
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model


NUM_ARTISTS = 10

# get directories
main_direc = os.getcwd()
images_dir = os.path.join(main_direc, 'data/images/images')

# csv location
artist_csv_loc = os.path.join(main_direc, 'data/artists.csv')


# Collecting Needed Images
artists = pd.read_csv(artist_csv_loc)

# Creating a dataframe with the top 10 artists by number of paintings
artists_sort = artists.sort_values(by=['paintings'], ascending=False)

artists_top = artists_sort.head(NUM_ARTISTS) # need to add 1 so 10 classes are read in
print(artists_top)

# Images DataFrame
artists_top_name = artists_top['name'].str.replace(' ', '_').values

images_df = pd.DataFrame()
for name in artists_top_name:
    images_df = pd.concat([images_df, pd.DataFrame(data={'Path': glob.glob('data/images/images/' + name + '/*'), 'Name': name})], ignore_index=True)

print(images_df)

# Create Generator


BATCH_SIZE = 64

# image dimensions?
img_width, img_height = 277, 277

train_df = images_df.sample(frac=0.8, random_state=200)
test_df = images_df.drop(train_df.index)

if K.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

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
    top_model = Dense(500, activation='relu')(top_model)
    top_model = Dense(100, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)

    model = Model(inputs=conv_base.input, outputs=output_layer)

    model.compile(optimizer=optimizer,
                  loss=tf.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model

#step sizes:
steps_train = train_gen.n//train_gen.batch_size
steps_valid = valid_gen.n//valid_gen.batch_size
steps_test = test_gen.n//test_gen.batch_size


optimizer = tf.optimizers.Adam(learning_rate=0.001)
n_classes = 10
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
plt.savefig('john_model_accuracy.jpg')

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
plt.savefig('john_model_loss.jpg')


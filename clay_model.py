from numpy.lib.npyio import load
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import glob

def load_data(artists_csv_loc='data/artists.csv', images_dir='data/images/images'):
    artists = pd.read_csv(artists_csv_loc)
    print(artists.shape)
    # print(artists)

    # Creating a dataframe with the top 10 artists by number of paintings
    artists_top = artists.sort_values(by=['paintings'], ascending=False)
    print(artists_top)

    # Images
    artists_dir = os.listdir(images_dir) # Files are named after each artists

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

    


def create_model():
    pass


def train_model():
    pass


def main():
    load_data()

main()
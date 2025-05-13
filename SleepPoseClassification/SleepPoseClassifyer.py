import os
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datasets
import os
import cv2
import keras
from keras import ops


class SleepNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2, channel_3, fc1, num_classes):
        input_shape = (64, 64, 3)
        super(SleepNet, self).__init__()

        initializer = tf.initializers.VarianceScaling(scale=2.0)

        self.conv11 = tf.keras.layers.Conv2D(channel_1, (3, 3), activation='relu', padding='same',
                                             kernel_initializer=initializer)
        self.conv12 = tf.keras.layers.Conv2D(channel_1, (3, 3), activation='relu', padding='same',
                                             kernel_initializer=initializer)
        self.conv21 = tf.keras.layers.Conv2D(channel_2, (3, 3), activation='relu', padding='same',
                                             kernel_initializer=initializer)
        self.conv22 = tf.keras.layers.Conv2D(channel_2, (3, 3), activation='relu', padding='same',
                                             kernel_initializer=initializer)
        self.conv31 = tf.keras.layers.Conv2D(channel_3, (3, 3), activation='relu', padding='same',
                                             kernel_initializer=initializer)
        self.conv32 = tf.keras.layers.Conv2D(channel_3, (3, 3), activation='relu', padding='same',
                                             kernel_initializer=initializer)

        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()

        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        self.fc1 = tf.keras.layers.Dense(fc1, kernel_initializer=initializer, activation='relu')

        self.fc2 = tf.keras.layers.Dense(num_classes, kernel_initializer=initializer)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, input_tensor, training=False):
        x = self.conv11(input_tensor)
        x = self.conv12(x)

        x = self.maxpool(x)
        x = self.conv21(x)
        x = self.conv22(x)

        x = self.maxpool(x)
        x = self.conv31(x)
        x = self.conv32(x)

        x = self.maxpool(x)
        x = self.flatten(x)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class SleepPoseClassifyer:
    def __init__(self):

        channel_1 = 32
        channel_2 = 64
        channel_3 = 128
        fc1 = 512
        num_classes = 3

        test_image = [np.ones((64,64,3), dtype=np.float32)]
        test_labels = [np.zeros((1), dtype=np.float32)]

        self.model = SleepNet(channel_1, channel_2, channel_3, fc1, num_classes)
        self.model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=[tf.keras.metrics.sparse_categorical_accuracy])
        self.model.fit(np.array(test_image), np.array(test_labels), epochs=1, batch_size=1)

        self.model.load_weights('SleepPoseClassification/SleepNetModelv2_95.weights.h5')


    def batch_classify(self, images):
        return np.argmax(self.model.predict(np.array(images,dtype=np.float32)), axis=-1)


if __name__ == '__main__':
    image1 = cv2.imread('Dataset/Test/left_log/777.png')
    image2 = cv2.imread('Dataset/Test/right_log/777.png')
    image3 = cv2.imread('Dataset/Test/supine/777.png')

    model = SleepPoseClassifyer()
    identity_matr = model.batch_classify([image2,image1,image3])

    print(np.round(identity_matr,0))

# from tensorflow_docs.vis import embed
from tensorflow import keras
from keras.utils import plot_model
from imutils import paths
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, GRU, Dense, ReLU, GlobalAveragePooling2D,TimeDistributed

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import csv

from flirvideo import FlirVideo

R = 120
C = 128
BATCH_SIZE = 64
EPOCHS = 1500

# MAX_SEQ_LENGTH = 5
MAX_SEQ_LENGTH = 1500
TRAIN_RANDOM_AUG = 4

MAX_THREAD_WORKERS = 12

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# print(f"Total videos for training: {len(train_df)}")
# print(f"Total videos for testing: {len(test_df)}")

# train_df.sample(10)

def load_video(Temp, max_frames=0, resize=(R, C)):
    min = Temp.min()
    max = Temp.max()

    Temp = 255*(Temp-min)/(max-min)

    frames = []
    len = Temp.shape[2]
    
    for i in range(len):
        size = Temp.shape

        if size[0] < resize[0]:
            d1 = np.int16(np.floor((resize[0]-size[0])/2))
            d2 = np.int16(resize[0]-size[0]-d1)

            T = np.append(np.zeros((d1, size[1])), np.append(Temp[:,:, i], np.zeros((d2, size[1])),axis=0),axis=0)
        else:
            d1 = np.int16(np.floor((size[0]-resize[0])/2))
            T = Temp[d1:d1+resize[0], :, i]

        if size[1] < resize[1]:
            d1 = np.int16(np.floor((resize[1]-size[1])/2))
            d2 = np.int16(resize[1]-size[1]-d1)

            T = np.append(np.zeros((resize[0],d1)), np.append(T, np.zeros((resize[0],d2)),axis=1),axis=1)
        else:
            d1 = np.int16(np.floor((size[1]-resize[1])/2))
            T = T[:, d1:d1+resize[1]]


        # frame = np.dstack((T[:,:], np.zeros(resize), np.zeros(resize)))
        frame = np.dstack((T[:,:], T[:,:], T[:,:]))

        frames.append(frame)

    return np.array(frames)

def prepare_video(fv, randomize, frame_features, nugDiams, nugDiam, CBDiams, CBDiam, videoRep, idx, reps):
    # Gather all its frames and add a batch dimension.
    if randomize:
        frames = load_video(fv.generateRandomData())
    else:
        frames = load_video(fv.saveTemp())

    if frames.shape[0]>MAX_SEQ_LENGTH:
        frames = frames[:MAX_SEQ_LENGTH, :]

    if frames.shape[0]<MAX_SEQ_LENGTH:
         frames = np.concatenate((frames, np.zeros((MAX_SEQ_LENGTH-frames.shape[0], R,C,3))),0)
    
    frame_features[idx*reps+videoRep,] = frames

    nugDiams[idx*reps+videoRep] = nugDiam
    CBDiams[idx*reps+videoRep] = CBDiam


def prepare_all_videos(df, randomize, reps):
    num_samples = len(df)*reps
    video_paths = df["fileName"].values.tolist()

    # diams = df["diam"].values

    # `frame_features` are what we will feed to our sequence model.
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, R, C, 3, ), dtype="float32"
    )
    nugDiams = np.zeros(shape=(num_samples,1), dtype="float32")
    CBDiams = np.zeros(shape=(num_samples,1), dtype="float32")
 
    fv = []

    with ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS) as executor:
        # For each video.
        for idx, path in enumerate(video_paths):
            fv.append(FlirVideo(path))
            # print("Preparing idx = " + str(idx))
            # print("Cut range: " + str(fv[idx].findExcitmentPeriod(200)))
            fv[idx].videoCut(fv[idx].findExcitmentPeriod(200))

            for videoRep in range(reps):
                executor.submit(prepare_video, fv[idx], randomize, frame_features, nugDiams, df["Nugget"].values[idx], CBDiams, df["CB"].values[idx], videoRep, idx, reps)
                # prepare_video(fv[idx], randomize, frame_features, nugDiams, df["Nugget"].values[idx], CBDiams, df["CB"].values[idx], videoRep, idx, reps)

    fv = []
    return frame_features, nugDiams, CBDiams

train_data, train_nugDiams, train_CBDiams = prepare_all_videos(train_df, True, TRAIN_RANDOM_AUG)
print("Alive")
test_data, test_nugDiams, test_CBDiams = prepare_all_videos(test_df, False, 1)

maxNugDiam = test_nugDiams.max()
minNugDiam = test_nugDiams.min()
train_nugDiams = (train_nugDiams-minNugDiam)/(maxNugDiam-minNugDiam)
test_nugDiams = (test_nugDiams-minNugDiam)/(maxNugDiam-minNugDiam)

maxCBDiam = test_CBDiams.max()
minCBDiam = test_CBDiams.min()
train_CBDiams = (train_CBDiams-minCBDiam)/(maxCBDiam-minCBDiam)
test_CBDiams = (test_CBDiams-minCBDiam)/(maxCBDiam-minCBDiam)

print(f"Frame features in train set: {train_data.shape}")

def get_sequence_model():
    video_input = keras.Input(shape=(MAX_SEQ_LENGTH, R, C, 3))

    imagenet_model= keras.applications.vgg16.VGG16(input_shape=(R,C,3),weights='imagenet', include_top=False)

    imagenet_layers = imagenet_model.layers
    for layer in imagenet_layers:
        layer.trainable = False
    imagenet = keras.Model(imagenet_model.input, imagenet_layers[-1].output)

    x = TimeDistributed(imagenet)(video_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    x = GRU(128, activation='relu', return_sequences = True)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = GRU(64, activation='relu', return_sequences = False)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    out1 = keras.layers.Dense(1, activation="linear", name='Nugget')(x)
    out2 = keras.layers.Dense(1, activation="linear", name='CB')(x)

    rnn_model = keras.Model(video_input, out1)

    rnn_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return rnn_model

def plot_loss(history, fileName):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(fileName)
    plt.close()

def run_experiment():
    #  Early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    filepath = "./video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()

    # print(seq_model.summary())
    plot_model(seq_model, to_file='model.png', show_shapes=True, show_layer_names=True)
    history = seq_model.fit(
        train_data,
        train_nugDiams,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
        validation_data=(test_data, test_nugDiams)
    )

    print('First Training completed')

    #print metrics
    # Print loss and validation loss
    print('Loss:', history.history['loss'])
    print('Validation loss:', history.history['val_loss'])

    plot_loss(history, 'history1.pdf')
    seq_model.save('model.h5')

    # history = []

    seq_model.load_weights(filepath)
    # _, mse = seq_model.evaluate(test_data[0], [test_nugDiams, test_CBDiams])
    # print(f"Validation MAE {mse}")

    for layer in seq_model.layers:
        layer.trainable = True

    seq_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse')  # Lower learning rate

    history = seq_model.fit(
        train_data,
        train_nugDiams,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
        validation_data=(test_data, test_nugDiams)
    )

    print("Second training completed")
    plot_loss(history, 'history2.pdf')
    seq_model.save('modelDef.h5')
    seq_model.load_weights(filepath)

    return history, seq_model


_, sequence_model = run_experiment()

def sequence_prediction(idx):
    video = test_data[idx,:]
    diam = sequence_model.predict(video)
    # print(f"Estimated diams: {diams[0]}, {diams[1]}")
    return diam

est_nugDiam = np.zeros(len(test_df))
est_CBDiam = np.zeros(len(test_df))
nugDiams = test_df["Nugget"].values
CBDiams = test_df["CB"].values

with open('out.csv', 'w', encoding='UTF8') as fd:
    writer = csv.writer(fd)

    for idx, path in enumerate(test_df["fileName"].values.tolist()):
        # print(f"Test video path: {path}")
        est_nugDiam[idx] = sequence_prediction(idx)

        writer.writerow([path, nugDiams[idx], CBDiams[idx], est_nugDiam[idx]*(maxNugDiam-minNugDiam)+minNugDiam])

plt.clf()
plt.plot(nugDiams, est_nugDiam*(maxNugDiam-minNugDiam)+minNugDiam, 'r+')
plt.plot(nugDiams, nugDiams)
plt.savefig('validation_nugget.pdf')

# plt.clf()
# plt.plot(CBDiams, est_CBDiam*normCBDiam, 'r+')
# plt.plot(CBDiams, CBDiams)
# plt.savefig('validation_CB.pdf')

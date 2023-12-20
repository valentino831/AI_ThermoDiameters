# from tensorflow_docs.vis import embed
from tensorflow import keras
from keras.utils import plot_model
from imutils import paths

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
MAX_SEQ_LENGTH = 800
TRAIN_RANDOM_AUG = 5
NUM_FEATURES = 2048

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


        frame = np.dstack((T[:,:], np.zeros(resize), np.zeros(resize)))

        frames.append(frame)

    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(R, C, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((R, C, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def prepare_video(fv, randomize, frame_features, frame_masks, nugDiams, nugDiam, CBDiams, CBDiam, videoRep, idx, reps):
    # Gather all its frames and add a batch dimension.
    if randomize:
        frames = load_video(fv.generateRandomData())
    else:
        frames = load_video(fv.saveTemp())

    frames = frames[None, ...]

    # Initialize placeholders to store the masks and features of the current video.
    temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    temp_frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    frame_features[idx*reps+videoRep,] = temp_frame_features.squeeze()
    frame_masks[idx*reps+videoRep,] = temp_frame_mask.squeeze()

    nugDiams[idx*reps+videoRep] = nugDiam
    CBDiams[idx*reps+videoRep] = CBDiam


def prepare_all_videos(df, randomize, reps):
    num_samples = len(df)*reps
    video_paths = df["fileName"].values.tolist()

    # diams = df["diam"].values

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
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
                executor.submit(prepare_video, fv[idx], randomize, frame_features, frame_masks, nugDiams, df["Nugget"].values[idx], CBDiams, df["CB"].values[idx], videoRep, idx, reps)

    fv = []
    return (frame_features, frame_masks), nugDiams, CBDiams

train_data, train_nugDiams, train_CBDiams = prepare_all_videos(train_df, True, TRAIN_RANDOM_AUG)
test_data, test_nugDiams, test_CBDiams = prepare_all_videos(test_df, False, 1)

# print(f"Frame features in train set: {train_data[0].shape}")
# print(f"Frame masks in train set: {train_data[1].shape}")

def get_sequence_model():
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    # x = keras.layers.GRU(16, return_sequences=True)(
    #     frame_features_input, mask=mask_input
    # )
    # x = keras.layers.GRU(8)(x)
    # x = keras.layers.Dropout(0.6)(x)
    # x = keras.layers.Dense(8, activation="relu")(x)
    # output = keras.layers.Dense(1, activation="relu")(x)


    # Questo mi piace
    # x = keras.layers.GRU(24, return_sequences=True)(
    #     frame_features_input, mask=mask_input
    # )
    # x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.GRU(12)(x)
    # x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.Dense(8, activation="relu")(x)
    # x = keras.layers.Dropout(0.5)(x)

    # x = keras.layers.GRU(32, return_sequences=True)(
    #     frame_features_input, mask=mask_input
    # )

    x = keras.layers.GRU(32, return_sequences=True)(
        frame_features_input)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.GRU(16)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(4, activation="relu")(x)

    out1 = keras.layers.Dense(1, activation="linear", name='Nugget')(x)
    out2 = keras.layers.Dense(1, activation="linear", name='CB')(x)

    rnn_model = keras.Model(frame_features_input, [out1, out2])

    rnn_model.compile(
        loss="mean_absolute_error", optimizer="adam", metrics=["mae", "mse"]
    )
    return rnn_model

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('history.pdf')

def run_experiment():
    filepath = "./video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    # print(seq_model.summary())
    # plot_model(seq_model, to_file='model.png', show_shapes=True, show_layer_names=True)
    history = seq_model.fit(
        train_data[0],
        [train_nugDiams, train_CBDiams],
        epochs=EPOCHS,
        callbacks=[checkpoint],
        validation_data=(test_data[0], [test_nugDiams, test_CBDiams])
    )

    plot_loss(history)
    seq_model.save('model.h5')

    # history = []

    seq_model.load_weights(filepath)
    # _, mse = seq_model.evaluate(test_data[0], [test_nugDiams, test_CBDiams])
    # print(f"Validation MAE {mse}")

    return history, seq_model


_, sequence_model = run_experiment()

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    fv = FlirVideo(path)
    fv.videoCut(fv.findExcitmentPeriod(200))
    frames = load_video(fv.saveTemp())
    frame_features, frame_mask = prepare_single_video(frames)
    diams = sequence_model.predict([frame_features])
    # print(f"Estimated diams: {diams[0]}, {diams[1]}")
    return diams[0], diams[1]

est_nugDiam = np.zeros(len(test_df))
est_CBDiam = np.zeros(len(test_df))
nugDiams = test_df["Nugget"].values
CBDiams = test_df["CB"].values

with open('out.csv', 'w', encoding='UTF8') as fd:
    writer = csv.writer(fd)

    for idx, path in enumerate(test_df["fileName"].values.tolist()):
        # print(f"Test video path: {path}")
        est_nugDiam[idx], est_CBDiam[idx] = sequence_prediction(path)

        writer.writerow([path, nugDiams[idx], CBDiams[idx], est_nugDiam[idx], est_CBDiam[idx]])

plt.clf()
plt.plot(nugDiams, est_nugDiam, 'r+')
plt.plot(nugDiams, nugDiams)
plt.savefig('validation_nugget.pdf')

plt.clf()
plt.plot(CBDiams, est_CBDiam, 'r+')
plt.plot(CBDiams, CBDiams)
plt.savefig('validation_CB.pdf')

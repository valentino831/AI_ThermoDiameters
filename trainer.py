from tensorflow_docs.vis import embed
from tensorflow import keras
from keras.utils import plot_model
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

from flirvideo import FlirVideo

R = 136
C = 144
BATCH_SIZE = 64
EPOCHS = 1500

# MAX_SEQ_LENGTH = 5
MAX_SEQ_LENGTH = 140
NUM_FEATURES = 2048

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

# train_df.sample(10)

def load_video(Temp, max_frames=0, resize=(R, C)):
    min = Temp.min()
    max = Temp.max()

    Temp = 255*(Temp-min)/(max-min)

    frames = []
    len = Temp.shape[2]
    
    for i in range(len):
        frame = np.dstack((Temp[:,:,i], 0*Temp[:,:,i], 0*Temp[:,:,i]))

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
    diams = np.zeros(shape=(num_samples,1), dtype="float32")
 
    # For each video.
    for idx, path in enumerate(video_paths):
        fv = FlirVideo(path)
        fv.videoCut(fv.findExcitmentPeriod(100))

        for videoRep in range(reps):
            # Gather all its frames and add a batch dimension.
            if randomize:
                frames = load_video(fv.generateRandomData())
            else:
                frames = load_video(fv.saveTemp())

            frames = frames[None, ...]

            # Initialize placeholders to store the masks and features of the current video.
            temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
            temp_frame_features = np.zeros(
                shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
            )

            # Extract features from the frames of the current video.
            for i, batch in enumerate(frames):
                video_length = batch.shape[0]
                length = min(MAX_SEQ_LENGTH, video_length)
                for j in range(length):
                    temp_frame_features[i, j, :] = feature_extractor.predict(
                        batch[None, j, :]
                    )
                temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

            frame_features[idx*reps+videoRep,] = temp_frame_features.squeeze()
            frame_masks[idx*reps+videoRep,] = temp_frame_mask.squeeze()

            diams[idx*reps+videoRep] = df["diam"].values[idx]

        fv = []

    return (frame_features, frame_masks), diams

train_data, train_diams = prepare_all_videos(train_df, True, 20)
test_data, test_labels = prepare_all_videos(test_df, False, 1)

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

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

    x = keras.layers.GRU(24, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.Dropout(0.6)(x)
    x = keras.layers.GRU(18)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    output = keras.layers.Dense(1, activation="relu")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="mean_absolute_error", optimizer="adam", metrics="mean_absolute_error"
    )
    return rnn_model

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [diam]')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiment():
    filepath = "./video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    print(seq_model.summary())
    # plot_model(seq_model, to_file='model.png', show_shapes=True, show_layer_names=True)
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_diams,
        validation_split=0.2,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    plot_loss(history)
    seq_model.save('model.h5')

    # history = []

    seq_model.load_weights(filepath)
    _, mse = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Validation MAE {mse}")

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
    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    diam_est = sequence_model.predict([frame_features, frame_mask])[0]

    print(f"Estimated diam: {diam_est}")
    return diam_est

exit()

est_diam = np.zeros(len(test_df))
diams = test_df["diam"].values

for idx, path in enumerate(test_df["fileName"].values.tolist()):
    print(f"Test video path: {path}")
    est_diam[idx] = sequence_prediction(path)

plt.plot(diams, est_diam, 'r+')
plt.plot(diams, diams)
plt.show()

print(np.sum((est_diam-diams)**2))

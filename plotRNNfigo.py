#questo codice plotta lo schema della RNN keras in analisi.py
#     model = keras.Model([frame_features_input, mask_input], output)
#     plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
#     #     epochs=100,
#     #     callbacks=[checkpoint],
#     # )   
#

#installare graphviz è un casino, ma alla fine ci sono riuscito
# credo che se usi sto comando te la sbrighi e lo installa nel virtual environment
#conda install -c conda-forge python-graphviz
#conda list | findstr "graphviz"
#su vscode non va su spyfer si

import pydot
pydot.find_graphviz = lambda: {'dot': 'C:/Users/lucsa/.conda/envs/ThermalAI/Library/bin/dot.exe'}



from tensorflow import keras
from keras.utils.vis_utils import plot_model

R = 136
C = 144
MAX_SEQ_LENGTH = 140
NUM_FEATURES = 2048

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

def get_sequence_model():
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
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
    return rnn_model

model = get_sequence_model()
plot_model(model, to_file='simplified_model_diagram.png', show_shapes=True, show_layer_names=True)
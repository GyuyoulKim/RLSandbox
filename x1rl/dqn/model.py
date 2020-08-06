from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def create_q_model(input_shape, action_space):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=input_shape)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(action_space, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

def create_duel_q_model(input_shape, action_space):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=input_shape)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    layer6 = layers.Dense(256, activation="relu")(layer5)
    layer7 = layers.Dense(64, activation="relu")(layer6)

    state_value = layers.Dense(1, activation='linear')(layer7)
    state_value = layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

    action_advantage = layers.Dense(action_space, activation='linear')(layer7)
    action_advantage = layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

    action = layers.Add()([state_value, action_advantage])

    return keras.Model(inputs=inputs, outputs=action)

def summarize_network():
    model = create_q_model((84,84,4), 4)
    model.summary()

    duel_model = create_duel_q_model((84,84,4), 4)
    duel_model.summary()

if __name__ == '__main__':
    summarize_network()
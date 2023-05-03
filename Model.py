from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv1D, Bidirectional,Flatten, Reshape,GRU, Input
from keras.models import Sequential, Model
from Spatial import spatial_attention






def create_model(time_stamp, model_name):

    if model_name == 'GRU':
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(16, 4, activation="relu", padding="same", strides=1, input_shape=(time_stamp, 1)))
        model.add(Conv1D(8, 4, activation="relu", padding="same", strides=1))

        # Bi-directional LSTMs
        model.add(Bidirectional(GRU(96, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(Bidirectional(GRU(96, return_sequences=False, stateful=False), merge_mode='concat'))

        # Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='linear'))

    if model_name == 'SGRU':
        input_tensor = Input(shape=(time_stamp,))
        inp = Reshape((time_stamp, 1), )(input_tensor)

        d1 = Dense(32, activation='relu', name='dense1')(inp)
        # Bi-directional GRUs
        b1 = Bidirectional(
            GRU(units=96, activation='tanh', recurrent_activation='sigmoid', stateful=False, return_sequences=True))(d1)
        b2 = Bidirectional(
            GRU(units=96, activation='tanh', recurrent_activation='sigmoid', stateful=False, return_sequences=True))(b1)

        s1 = spatial_attention(b2, time_stamp)

        f1 = Flatten()(s1)

        d = Dense(128, activation='relu', name='dense')(f1)

        d_out = Dense(1, activation='linear', name='uscita')(d)

        model = Model(inputs=input_tensor, outputs=d_out)

        return model

    if model_name == 'STUDENT':
        model = Sequential()

        model.add(Dense(60, activation='relu',input_shape=(time_stamp, 1)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='linear'))

    return model



from tensorflow.keras.layers import GlobalAveragePooling2D,AveragePooling2D, MaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Conv1D
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid



def spatial_attention(input_feature,window_length):
    kernel_size = 7

    cbam_feature = Reshape((window_length,192,1), )(input_feature)
    avg_pool = AveragePooling2D(pool_size=(1,2),padding = "same",data_format='channels_last')(cbam_feature)

    max_pool = MaxPooling2D(pool_size=(1,2),padding = "same",data_format='channels_last')(cbam_feature)

    concat = Concatenate(axis=2)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          data_format='channels_last',padding = "same"
                          )(concat)
    mul = Reshape((window_length,192), )(cbam_feature)
    out = multiply([input_feature, mul])

    return out
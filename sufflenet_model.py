import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add, AvgPool2D, 
                                     Concatenate, GlobalAvgPool2D, Dense, Input, Lambda, Dropout, 
                                     MaxPool2D, Reshape, Permute)
from tensorflow.keras.regularizers import l2

class ShuffleNetV2:
    def __init__(self, input_shape=(224, 224, 3), num_classes=None, repetitions=(4, 8, 4), 
                 initial_channels=512, groups=8, dropout_rate=0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.repetitions = repetitions
        self.initial_channels = initial_channels
        self.groups = groups
        self.dropout_rate = dropout_rate

    def gconv(self, tensor, channels, groups):
        input_ch = tensor.get_shape().as_list()[-1]
        group_ch = input_ch // groups
        output_ch = channels // groups
        groups_list = []

        for i in range(groups):
            group_tensor = Lambda(lambda x: x[:, :, :, i * group_ch: (i + 1) * group_ch])(tensor)
            group_tensor = Conv2D(output_ch, 1)(group_tensor)
            groups_list.append(group_tensor)

        output = Concatenate()(groups_list)
        return output

    def channel_shuffle(self, x, groups):
        _, width, height, channels = x.get_shape().as_list()
        group_ch = channels // groups

        x = Reshape([width, height, group_ch, groups])(x)
        x = Permute([1, 2, 4, 3])(x)
        x = Reshape([width, height, channels])(x)
        return x

    def shufflenet_block(self, tensor, channels, strides, groups):
        x = self.gconv(tensor, channels=channels // 4, groups=groups)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = self.channel_shuffle(x, groups)
        x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)

        if strides == 2:
            channels = channels - tensor.get_shape().as_list()[-1]
        x = self.gconv(x, channels=channels, groups=groups)
        x = BatchNormalization()(x)

        if strides == 1:
            x = Add()([tensor, x])
        else:
            avg = AvgPool2D(pool_size=3, strides=2, padding='same')(tensor)
            x = Concatenate()([avg, x])

        output = ReLU()(x)
        return output

    def stage(self, x, channels, repetitions, groups):
        x = self.shufflenet_block(x, channels=channels, strides=2, groups=groups)
        for i in range(repetitions):
            x = self.shufflenet_block(x, channels=channels, strides=1, groups=groups)
        return x

    def build_model(self):
        input = Input(self.input_shape)
        x = Conv2D(filters=24, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2(0.001))(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        for i, reps in enumerate(self.repetitions):
            channels = self.initial_channels * (2 ** i)
            x = self.stage(x, channels, reps, self.groups)

        x = GlobalAvgPool2D()(x)
        x = Dropout(self.dropout_rate)(x)

        output = Dense(self.num_classes, activation='softmax')(x)

        model = tf.keras.Model(input, output)
        return model
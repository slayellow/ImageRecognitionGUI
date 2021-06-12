import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, GlobalAvgPool2D, BatchNormalization, Dense, \
    Input, Dropout, Flatten, DepthwiseConv2D


def set_conv(channel, kernel, strides=(1,1), name=None, padding='same', bias=False):
    if name is None:
        conv = Conv2D(channel, kernel, strides=strides, padding=padding, use_bias=bias)
    else:
        conv = Conv2D(channel, kernel, strides=strides, name=name, padding=padding, use_bias=bias)
    return conv


def set_depthwise_conv(kernel, strides=(1,1), name=None, bias=False, padding='same', data_format='channels_last'):
    if name is None:
        conv = DepthwiseConv2D(kernel_size=kernel, strides=strides, use_bias=bias, padding=padding, data_format=data_format)
    else:
        conv = DepthwiseConv2D(kernel_size=kernel, strides=strides, use_bias=bias, padding=padding, name=name, data_format=data_format)
    return conv


def set_flatten():
    return Flatten()


def set_batch_normalization(name=None, momentum=0.9, epsilon=1e-5):
    if name is None:
        bn = BatchNormalization(momentum=momentum, epsilon=epsilon)
    else:
        bn = BatchNormalization(name=name, momentum=momentum, epsilon=epsilon)
    return bn


def set_relu(layer):
    return tf.nn.relu(layer)


def set_relu6(layer):
    return tf.nn.relu6(layer)


def set_avg_pool(layer, ksize=(2,2), strides=(2,2), padding='SAME'):
    return tf.nn.avg_pool2d(layer, ksize=ksize, strides=strides, padding=padding)


def set_max_pool(layer, ksize=(2,2), strides=(2,2), padding='SAME'):
    return tf.nn.max_pool2d(layer, ksize=ksize, strides=strides, padding=padding)


def set_dense(channel, name=None, activation='relu', use_bias=False):
    if name is None:
        dense = Dense(channel, activation=activation, use_bias=use_bias)
    else:
        dense = Dense(channel, name=name, activation=activation, use_bias=use_bias)
    return dense


def set_dropout(rate=0.5):
    return Dropout(rate=rate)


def set_global_average_pooling():
    return GlobalAvgPool2D()


def summary(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary


def build(model, input_shape):
    model.build(input_shape=input_shape)

def call(model, input_shape, is_bn=False):
    if is_bn is False:
        model.call(Input(shape=input_shape), training=False)
    else:
        model.call(Input(shape=input_shape), training=True)


def save_weight_parameter(model, name, ext='h5'):
    if ext == 'h5':
        model.save_weights(name+'.'+ext, save_format=ext)


def load_weight_parameter(model, name, ext='h5'):
    if ext == 'h5':
        model.load_weights(name+'.'+ext)


def loss_cross_entropy(y_true, y_pred, label_smoothing=0.0):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy


def get_accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return accuracy


def get_correct_number(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return correct_num


def l2_loss(model, weights=1e-4):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


def set_SGD(learning_rate, momentum=0.9):
    return optimizers.SGD(learning_rate=learning_rate, momentum=momentum)


def set_Adam(learning_rate):
    return optimizers.Adam(learning_rate)


def set_Adagrad(learning_rate):
    return optimizers.Adagrad(learning_rate=learning_rate)


def set_RMSProp(learning_rate):
    return optimizers.RMSprop(learning_rate=learning_rate)

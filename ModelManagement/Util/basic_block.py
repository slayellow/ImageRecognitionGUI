from ModelManagement.Util.keras_util import *


# ResNet-18, ResNet-34을 위한 Block
class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, channels, strides=(1, 1), **kwargs):
        self.strides = strides

        # strides가 (1,1)이 아닌경우, 채널의 갯수가 반으로 줄어드는 경우이므로, Shortcut Connection을 위한 Conv 진행
        if strides != (1, 1):
            self.shortcut = set_conv(channels, (1, 1), strides=strides, name='shortcut')

        self.conv_0 = set_conv(channels, (3, 3), strides=strides, name='conv_0')
        self.conv_1 = set_conv(channels, (3, 3), name='conv_1')
        self.bn_0 = set_batch_normalization(name='bn_0')
        self.bn_1 = set_batch_normalization(name='bn_1')
        super(BasicBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)      # Training 진행 시 True
        net = tf.nn.relu(net)

        if self.strides != (1, 1):
            shortcut = self.shortcut(net)
        else:
            shortcut = inputs

        net = self.conv_0(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv_1(net)

        output = net + shortcut
        return output

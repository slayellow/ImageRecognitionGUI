from ModelManagement.Util.keras_util import *

import warnings


class VGGNet(tf.keras.models.Model):
    # VGGNet-16     : Block             :   [2, 2, 3, 3, 3]
    # VGGNet-19     : Block             :   [2, 2, 4, 4, 4]

    # Channel       : [64, 128, 256, 512, 512]
    def __init__(self, layer_num, classes, **kwargs):
        super(VGGNet, self).__init__(**kwargs)

        self.model_name = 'VGGNet_{}'.format(layer_num)
        self.layer_num = layer_num

        # ResNet의 기본 구성
        blocks = {16: (2, 2, 3, 3, 3),
                  19: (3, 4, 6, 3)}
        channels = (64, 128, 256, 512, 512)

        if layer_num is 16 or layer_num is 19:
            self.conv0_0 = set_conv(channels[0], (3, 3), name='conv0_0')
            self.conv0_1 = set_conv(channels[0], (3, 3), name='conv0_1')
            self.bn0_0 = set_batch_normalization(name='bn0_0')
            self.bn0_1 = set_batch_normalization(name='bn0_1')
            self.dropout0 = set_dropout(rate=0.5)

            self.conv1_0 = set_conv(channels[1], (3, 3), name='conv1_0')
            self.conv1_1 = set_conv(channels[1], (3, 3), name='conv1_1')
            self.bn1_0 = set_batch_normalization(name='bn1_0')
            self.bn1_1 = set_batch_normalization(name='bn1_1')
            self.dropout1 = set_dropout(rate=0.5)

            self.conv2_0 = set_conv(channels[2], (3, 3), name='conv2_0')
            self.conv2_1 = set_conv(channels[2], (3, 3), name='conv2_1')
            self.conv2_2 = set_conv(channels[2], (3, 3), name='conv2_2')
            self.bn2_0 = set_batch_normalization(name='bn2_0')
            self.bn2_1 = set_batch_normalization(name='bn2_1')
            self.bn2_2 = set_batch_normalization(name='bn2_2')
            self.dropout2 = set_dropout(rate=0.5)
            self.dropout3 = set_dropout(rate=0.5)
            if layer_num is 19:
                self.conv2_3 = set_conv(channels[2], (3,3), name='conv2_3')
                self.bn2_3 = set_batch_normalization(name='bn2_3')

            self.conv3_0 = set_conv(channels[3], (3, 3), name='conv3_0')
            self.conv3_1 = set_conv(channels[3], (3, 3), name='conv3_1')
            self.conv3_2 = set_conv(channels[3], (3, 3), name='conv3_2')
            self.bn3_0 = set_batch_normalization(name='bn3_0')
            self.bn3_1 = set_batch_normalization(name='bn3_1')
            self.bn3_2 = set_batch_normalization(name='bn3_2')
            self.dropout4 = set_dropout(rate=0.5)
            self.dropout5 = set_dropout(rate=0.5)
            if layer_num is 19:
                self.conv3_3 = set_conv(channels[3], (3,3), name='conv3_3')
                self.bn3_3 = set_batch_normalization(name='bn3_3')

            self.conv4_0 = set_conv(channels[4], (3, 3), name='conv4_0')
            self.conv4_1 = set_conv(channels[4], (3, 3), name='conv4_1')
            self.conv4_2 = set_conv(channels[4], (3, 3), name='conv4_2')
            self.bn4_0 = set_batch_normalization(name='bn4_0')
            self.bn4_1 = set_batch_normalization(name='bn4_1')
            self.bn4_2 = set_batch_normalization(name='bn4_2')
            self.dropout6 = set_dropout(rate=0.5)
            self.dropout7 = set_dropout(rate=0.5)
            if layer_num is 19:
                self.conv4_3 = set_conv(channels[4], (3,3), name='conv4_3')
                self.bn4_3 = set_batch_normalization(name='bn4_3')
        else:
            warnings.warn("클래스가 구성하는 Layer 갯수와 맞지 않습니다.")
        self.flatten = set_flatten()
        self.fcl1 = set_dense(4096, name='fcl1')
        self.bn1 = set_batch_normalization(name='bn1')
        self.fcl2 = set_dense(4096, name='fcl2')
        self.bn2 = set_batch_normalization(name='bn2')
        self.fcl3 = set_dense(classes, name='fcl3', activation='softmax')

    def call(self, inputs, training):
        # --------------------------------------------- #
        net = self.conv0_0(inputs)
        net = self.bn0_0(net, training=training)
        net = set_relu(net)
        net = self.dropout0(net)

        net = self.conv0_1(net)
        net = self.bn0_1(net, training=training)
        net = set_relu(net)
        net = set_max_pool(net)
        # --------------------------------------------- #
        net = self.conv1_0(net)
        net = self.bn1_0(net, training=training)
        net = set_relu(net)
        net = self.dropout1(net)

        net = self.conv1_1(net)
        net = self.bn1_1(net, training=training)
        net = set_relu(net)
        net = set_max_pool(net)
        # -------------------------------------------- #
        net = self.conv2_0(net)
        net = self.bn2_0(net, training=training)
        net = set_relu(net)
        net = self.dropout2(net)

        net = self.conv2_1(net)
        net = self.bn2_1(net, training=training)
        net = set_relu(net)
        net = self.dropout3(net)

        net = self.conv2_2(net)
        net = self.bn2_2(net, training=training)
        net = set_relu(net)

        if self.layer_num is 19:
            net = self.conv2_3(net)
            net = self.bn2_3(net, training=training)
            net = set_relu(net)

        net = set_max_pool(net)
        # -------------------------------------------- #
        net = self.conv3_0(net)
        net = self.bn3_0(net, training=training)
        net = set_relu(net)
        net = self.dropout4(net)

        net = self.conv3_1(net)
        net = self.bn3_1(net, training=training)
        net = set_relu(net)
        net = self.dropout5(net)

        net = self.conv3_2(net)
        net = self.bn3_2(net, training=training)
        net = set_relu(net)

        if self.layer_num is 19:
            net = self.conv3_3(net)
            net = self.bn3_3(net, training=training)
            net = set_relu(net)

        net = set_max_pool(net)
        # -------------------------------------------- #
        net = self.conv4_0(net)
        net = self.bn4_0(net, training=training)
        net = set_relu(net)
        net = self.dropout6(net)

        net = self.conv4_1(net)
        net = self.bn4_1(net, training=training)
        net = set_relu(net)
        net = self.dropout7(net)

        net = self.conv4_2(net)
        net = self.bn4_2(net, training=training)
        net = set_relu(net)

        if self.layer_num is 19:
            net = self.conv4_3(net)
            net = self.bn4_3(net, training=training)
            net = set_relu(net)

        net = set_max_pool(net)
        # -------------------------------------------- #

        net = self.flatten(net)

        net = self.fcl1(net)
        net = self.bn1(net, training)
        net = set_relu(net)

        net = self.fcl2(net)
        net = self.bn2(net, training)
        net = set_relu(net)

        net = self.fcl3(net)

        return net

    def get_name(self):
        return self.model_name
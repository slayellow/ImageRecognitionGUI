from ModelManagement.KerasModel.ResNet import ResNet
from ModelManagement.KerasModel.VGGNet import VGGNet
from ModelManagement.KerasModel.MobileNet_V1 import MobileNet_V1
from ModelManagement.KerasModel.MobileNet_V2 import MobileNet_V2
from DataManagement.data_management import DataManagement
from ModelManagement.Util.keras_util import *
import os
from tqdm import tqdm
import numpy as np


class ModelManagement:

    def __init__(self):
        self.total_epoch = 0
        self.current_epoch = 0
        self.ext = None
        self.model = None
        self.state = 'Ready'
        self.summary = ''
        self.image_net_train = None
        self.image_net_validation = None
        self.image_net_test = None
        self.optimizer = None
        self.train_data_size = 0
        self.validation_data_size = 0
        self.test_data_size = 0
        self.batch_size = 0
        self.validation_batch_size = 0
        self.learning_rate = 1e-5
        self.train_idx = 0
        self.train_total_idx = 0
        self.train_loss = 0
        self.train_accuracy = 0
        self.train_validation_epoch = 0
        self.train_validation_total_epoch = 0
        self.validation_loss = 0
        self.validation_accuracy = 0
        self.test_batch_size = 0
        self.test_idx_current = 0
        self.test_idx_total = 0
        self.input_shape = 0
        self.test_result = []
        self.validation_result = []
        pass

    def print_state(self):
        print(self.state)

    @tf.function
    def validate_step(self, model, images, labels):
        prediction = model(images, training=False)
        cross_entropy = loss_cross_entropy(labels, prediction)
        prediction_label = tf.argmax(prediction, 1)
        return cross_entropy, prediction, prediction_label

    def validate_per_epoch(self, model):
        self.validation_result.clear()

        if self.image_net_validation is None:
            self.state = 'No Data!'
            return

        sum_ce = 0
        sum_correct_num = 0
        itr_per_epoch = int(self.validation_data_size / self.validation_batch_size)
        data_iterator = self.image_net_validation.get_data()
        self.state = "Validate!"
        for i in tqdm(range(itr_per_epoch)):
            images, labels, image_path = data_iterator.next()
            cross_entropy, prediction, prediction_label = self.validate_step(model, images, labels)
            correct_num = get_correct_number(labels, prediction)

            sum_ce += cross_entropy * self.validation_batch_size
            sum_correct_num += correct_num
            self.validation_result.append([image_path, prediction_label, np.argmax(labels, 1)])
        self.validation_loss = float(sum_ce / self.validation_data_size)
        self.validation_accuracy = float(sum_correct_num / self.validation_data_size)
        print('[Validation] cross entropy : {:.4f}, accuracy: {:.4f}'.format(sum_ce / self.validation_data_size,
                                                                             sum_correct_num / self.validation_data_size))

    def get_validation_result(self):
        validation_list = self.validation_result
        return validation_list

    @tf.function
    def test_step(self, model, images):
        prediction = model(images, training=False)
        prediction = tf.argmax(prediction, 1)
        return prediction

    # R-IR-SFR-003
    def test(self):
        self.test_result.clear()

        if self.image_net_test is None:
            self.state = 'No Data!'
            return

        # load pretrain
        if '{}.{}'.format(self.model.get_name(), self.ext) is not None:
            if os.path.isfile('{}.{}'.format(self.model.get_name(), self.ext)):
                load_weight_parameter(self.model, self.model.get_name(), self.ext)
                self.state = 'Load Pretrained Data'
            else:
                self.state = 'No Pretrained Data'

        self.state = "Testing"
        test_itration = int(self.test_data_size / self.test_batch_size)
        data_iterator = self.image_net_test.get_data()
        for i in tqdm(range(test_itration)):
            self.test_idx_current = int(i + 1)
            self.test_idx_total = int(test_itration)
            images, _, image_path = data_iterator.next()
            prediction = self.test_step(self.model, images)
            self.test_result.append([image_path, int(prediction)])

        self.state = 'Testing Finished!'

        return self.test_result

    @tf.function
    def train_step(self, model, images, labels, optimizer):
        with tf.GradientTape() as tape:
            prediction = model(images, training=True)
            cross_entropy = loss_cross_entropy(labels, prediction, label_smoothing=0.1)
            l2 = l2_loss(model)
            loss = cross_entropy + l2
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return cross_entropy, prediction

    def train_per_epoch(self, model, data_iterator, optimizer):
        itr_per_epoch = int(self.train_data_size / self.batch_size)
        sum_ce = 0
        sum_correct_num = 0

        for i in tqdm(range(itr_per_epoch)):
            images, labels, image_path = data_iterator.next()
            cross_entropy, prediction = self.train_step(model, images, labels, optimizer)
            correct_num = get_correct_number(labels, prediction)

            sum_ce += cross_entropy * self.batch_size
            sum_correct_num += correct_num

            self.train_idx = int(i + 1)
            self.train_total_idx = int(itr_per_epoch)
            self.train_loss = float(cross_entropy)
            self.train_accuracy = float(correct_num / self.batch_size)
            print('[Train] cross entropy loss: {:.4f}, accuracy: {:.4f}, l2 loss: {:.4f}'.format(cross_entropy,
                                                                                                 correct_num / self.batch_size,
                                                                                                 l2_loss(model)))

    def get_test_result(self):
        return self.test_idx_current, self.test_idx_total

    # R-IR-SFR-002
    def get_training_result(self):
        return self.train_idx, self.train_total_idx, self.train_loss, self.train_accuracy

    def get_epoch(self):
        return self.train_validation_epoch, self.train_validation_total_epoch

    # R-IR-SFR-005
    def get_validation_result(self):
        return self.validation_loss, self.validation_accuracy

    # R-IR-SFR-005
    def get_train_validation_rate(self):
        total = self.train_data_size + self.validation_data_size
        return self.train_data_size / total, self.validation_data_size / total

    # R-IR-SFR-001
    def train(self):
        if self.image_net_train is None:
            self.state = 'No Data!'
            return

        # load pretrain
        if '{}.{}'.format(self.model.get_name(), self.ext) is not None:
            if os.path.isfile('{}.{}'.format(self.model.get_name(), self.ext)):
                load_weight_parameter(self.model, self.model.get_name(), self.ext)
                self.state = 'Load Pretrained Data'
            else:
                self.state = 'No Pretrained Data'

        if self.total_epoch < 1:
            self.state = 'Train Error(Epoch)'
            return
        else:
            self.state = 'Training'
            for epoch in range(self.total_epoch):
                self.train_validation_epoch = epoch + 1
                self.train_validation_total_epoch = self.total_epoch
                self.current_epoch = epoch
                self.train_per_epoch(self.model, self.image_net_train.get_data(), self.optimizer)
                if self.image_net_validation is not None:
                    self.validate_per_epoch(self.model)
                save_weight_parameter(self.model, self.model.get_name(), ext=self.ext)

                # save intermediate results
                if self.total_epoch % 5 == 4:
                    os.system('cp {} {}_epoch_{}.h5'.format('{}.{}'.format(self.model.get_name(), self.ext),
                                                            '{}.{}'.format(self.model.get_name(), self.ext).split('.')[
                                                                0], epoch))
        self.state = 'Training Finished!'

    # R-IR-SFR-006
    def save_parameter(self, filepath, ext):
        file_path = filepath + '/' + self.model.get_name()
        save_weight_parameter(self.model, file_path, ext=ext)
        self.state = 'Save Success!'

    # R-IR-SFR-007
    def load_model(self, name):
        self.model = None
        if name is 'resnet_18':
            self.model = ResNet(18, 1000)
        elif name is 'resnet_34':
            self.model = ResNet(34, 1000)
        elif name is 'resnet_50':
            self.model = ResNet(50, 1000)
        elif name is 'resnet_101':
            self.model = ResNet(101, 1000)
        elif name is 'resnet_152':
            self.model = ResNet(152, 1000)
        elif name is 'vggnet_16':
            self.model = VGGNet(16, 1000)
        elif name is 'vggnet_19':
            self.model = VGGNet(19, 1000)
        elif name is 'mobilenet_v1':
            self.model = MobileNet_V1(1000, first_channel=32)
        elif name is 'mobilenet_v2':
            self.model = MobileNet_V2(1000)
        else:
            self.state = 'KerasModel is not detected!'
        self.state = 'KerasModel {} is loaded'.format(name)

    # R-IR-SFR-008 모델 구성
    def configure_model(self):
        pass

    # R-IR-SFR-009
    def check_model(self, mode='train'):
        if mode is 'train':
            build(self.model, (None,) + self.image_net_train.input_shape)
            call(self.model, self.image_net_train.input_shape, is_bn=True)
        elif mode is 'test':
            build(self.model, (None,) + self.image_net_test.input_shape)
            call(self.model, self.image_net_test.input_shape, is_bn=False)

        self.summary = summary(self.model)
        self.state = 'KerasModel Check Finish!'

    def load_test_dataset(self, data_path, label_file=None):
        self.image_net_test = DataManagement(data_path, list_file=label_file, mode=2)
        self.test_data_size = self.image_net_test.get_data_size()
        self.state = 'ImageNet Test Dataset Open!'

    def load_validation_dataset(self, data_path, label_file):
        self.image_net_validation = DataManagement(data_path, list_file=label_file, mode=1)
        self.validation_data_size = self.image_net_validation.get_data_size()
        self.state = 'ImageNet Validation Dataset Open!'

    def load_train_dataset(self, data_path, label_file):
        self.image_net_train = DataManagement(data_path, list_file=label_file, mode=0)
        self.train_data_size = self.image_net_train.get_data_size()
        self.state = 'ImageNet Training Dataset Open!'

    # R-IR-SFR-011
    def set_training_parameter(self, learning_rate, epoch, ext, batch_size, input_shape, augment, aspect, flip, hsv,
                               pca):
        self.learning_rate = learning_rate
        self.total_epoch = epoch
        self.ext = ext
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.image_net_train.set_batch_size(batch_size)
        self.image_net_train.set_input_shape(input_shape)
        self.image_net_train.set_augment(augment, aspect, flip, hsv, pca)
        self.state = 'Training Setting is Ready!'

    def set_testing_parameter(self, input_shape, batch_size=1):
        self.test_batch_size = batch_size
        self.image_net_test.set_input_shape(input_shape)
        self.image_net_test.set_batch_size(batch_size=batch_size)
        self.state = 'Testing Setting is Ready!'

    def set_validation_parameter(self, input_shape, batch_size=1):
        self.validation_batch_size = batch_size
        self.image_net_validation.set_input_shape(input_shape)
        self.image_net_validation.set_batch_size(batch_size=batch_size)
        self.state = 'Validation Setting is Ready!'

    # R-IR-SFR-010
    def set_optimizer(self, name):
        if name is 'sgd':
            self.optimizer = set_SGD(self.learning_rate)
        elif name is 'adam':
            self.optimizer = set_Adam(self.learning_rate)
        elif name is 'adagrad':
            self.optimizer = set_Adagrad(self.learning_rate)
        elif name is 'rmsprop':
            self.optimizer = set_RMSProp(self.learning_rate)
        self.state = 'Optimizer {} Open'.format(name)

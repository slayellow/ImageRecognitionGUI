from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from mainwindow_ui import Ui_MainWindow
import time

from ModelManagement.model_management import ModelManagement
from UtilityManagement.common_utils import *
from UtilityManagement.json import ParsingData
import threading


class MainWindow(Ui_MainWindow):
    def __init__(self, w):
        Ui_MainWindow.__init__(self)

        self.setupUi(w)
        set_gpu_setting()
        self.qwidget = QWidget()
        self.modelmanagement = ModelManagement()
        self.label_data = ParsingData('data/label_to_content.json')

        self.timer = None
        self.b_train_data = False
        self.b_validation_data = False
        self.b_test_data = False
        self.b_setting = False
        self.b_model = False
        self.b_optimizer = False
        self.test_result = []
        self.count = 0
        self.test_total = 0
        self.testtimer = None

        self.BTN_TRAIN.setEnabled(False)
        self.RB_DATA_AUGMENTATION_NO.setChecked(True)
        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(False)
        self.BTN_RESNET34.setEnabled(False)
        self.BTN_RESNET18.setEnabled(False)
        self.BTN_RESNET50.setEnabled(False)
        self.BTN_RESNET101.setEnabled(False)
        self.BTN_RESNET152.setEnabled(False)
        self.BTN_VGG16.setEnabled(False)
        self.BTN_VGG19.setEnabled(False)
        self.BTN_MOBILENET_V1.setEnabled(False)
        self.BTN_MOBILENET_V2.setEnabled(False)
        self.BTN_DENSENET121.setEnabled(False)
        self.BTN_DENSENET169.setEnabled(False)
        self.BTN_DENSENET201.setEnabled(False)
        self.BTN_SAVE_PARAMETER.setEnabled(False)
        self.BTN_INFERENCE.setEnabled(False)
        self.BTN_IMAGE_NEXT.setEnabled(False)
        self.BTN_IMAGE_PREV.setEnabled(False)
        self.TE_MODELCHECK.setFontPointSize(10)

        self.LB_STATE.setText(self.modelmanagement.state)

        self.BTN_LOAD_TRAINING_SET.clicked.connect(self.load_training_set)
        self.BTN_LOAD_VALIDATION_SET.clicked.connect(self.load_validation_set)
        self.BTN_LOAD_TESTING_SET.clicked.connect(self.load_testing_set)
        self.BTN_RESNET18.clicked.connect(self.load_resnet18)
        self.BTN_RESNET34.clicked.connect(self.load_resnet34)
        self.BTN_RESNET50.clicked.connect(self.load_resnet50)
        self.BTN_RESNET101.clicked.connect(self.load_resnet101)
        self.BTN_RESNET152.clicked.connect(self.load_resnet152)
        self.BTN_VGG16.clicked.connect(self.load_vggnet16)
        self.BTN_VGG19.clicked.connect(self.load_vggnet19)
        self.BTN_MOBILENET_V1.clicked.connect(self.load_mobilenet_v1)
        self.BTN_MOBILENET_V2.clicked.connect(self.load_mobilenet_v2)
        self.BTN_DENSENET121.clicked.connect(self.load_densenet121)
        self.BTN_DENSENET169.clicked.connect(self.load_densenet169)
        self.BTN_DENSENET201.clicked.connect(self.load_densenet201)
        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.clicked.connect(self.save_training_parameter)
        self.RB_DATA_AUGMENTATION_YES.clicked.connect(self.set_augmentation_radiobutton)
        self.RB_DATA_AUGMENTATION_NO.clicked.connect(self.set_augmentation_radiobutton)
        self.BTN_OPTIMIZER_SGD.clicked.connect(self.set_optimizer_sgd)
        self.BTN_OPTIMIZER_ADAM.clicked.connect(self.set_optimizer_adam)
        self.BTN_TRAIN.clicked.connect(self.train)
        self.BTN_SAVE_PARAMETER.clicked.connect(self.save_parameter)
        self.BTN_INFERENCE.clicked.connect(self.inference)
        self.BTN_IMAGE_NEXT.clicked.connect(self.next_image)
        self.BTN_IMAGE_PREV.clicked.connect(self.prev_image)
        self.BTN_OPTIMZER_ADAGRAD.clicked.connect(self.set_optimizer_adagrad)
        self.BTN_OPTIMZER_RMSPROP.clicked.connect(self.set_optimizer_rmsprop)

    def set_optimizer_adagrad(self):
        self.modelmanagement.set_optimizer('adagrad')
        self.b_optimizer = True
        self.LB_STATE.setText(self.modelmanagement.state)

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def set_optimizer_rmsprop(self):
        self.modelmanagement.set_optimizer('rmsprop')
        self.b_optimizer = True
        self.LB_STATE.setText(self.modelmanagement.state)

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def prev_image(self):
        self.count = self.count - 1
        if self.count < 0:
            self.count = 0
            return
        image_path = [tf.compat.as_str_any(tensor.numpy()) for tensor in self.test_result[self.count][0]]
        self.LB_IMAGE.setPixmap(QPixmap(image_path[0]))
        label = self.label_data.get_data(str(int(self.test_result[self.count][1])))
        self.LB_PREDICTION_LABEL.setText(label)

    def next_image(self):
        self.count = self.count + 1
        if self.count == self.test_total:
            self.count = self.count - 1
            return
        image_path = [tf.compat.as_str_any(tensor.numpy()) for tensor in self.test_result[self.count][0]]
        self.LB_IMAGE.setPixmap(QPixmap(image_path[0]))
        label = self.label_data.get_data(str(int(self.test_result[self.count][1])))
        self.LB_PREDICTION_LABEL.setText(label)

    def test_timeout(self):
        test_idx, self.test_total = self.modelmanagement.get_test_result()
        self.LB_TESTING_IDX_CURRENT.setNum(test_idx)
        self.LB_TESTING_IDX_TOTAL.setNum(self.test_total)
        self.testtimer = threading.Timer(0.5, self.test_timeout)
        self.testtimer.start()

    def on_test_thread(self):
        self.testtimer = threading.Timer(0.5, self.test_timeout)
        self.testtimer.start()
        self.test_result = self.modelmanagement.test()
        self.LB_STATE.setText(self.modelmanagement.state)
        time.sleep(1)
        self.testtimer.cancel()
        self.BTN_IMAGE_NEXT.setEnabled(True)
        self.BTN_IMAGE_PREV.setEnabled(True)
        image_path = [tf.compat.as_str_any(tensor.numpy()) for tensor in self.test_result[0][0]]
        self.LB_IMAGE.setPixmap(QPixmap(image_path[0]))
        label = self.label_data.get_data(str(int(self.test_result[0][1])))
        self.LB_PREDICTION_LABEL.setText(label)
        self.count = 0

    def inference(self):
        self.modelmanagement.check_model(mode='test')
        t = threading.Thread(target=self.on_test_thread)
        t.start()

    def save_parameter(self):
        save_path = QFileDialog.getExistingDirectory(self.qwidget, "Select Directory")
        if save_path is '':
            return
        self.modelmanagement.save_parameter(save_path, 'h5')

    def timeout(self):
        train_idx, train_total_idx, train_loss, train_accuracy = self.modelmanagement.get_training_result()
        epoch, total_epoch = self.modelmanagement.get_epoch()
        validation_loss, validation_accuracy = self.modelmanagement.get_validation_result()
        self.LB_TRAINING_EPOCH_CURRENT.setNum(epoch)
        self.LB_TRAINING_EPOCH_TOTAL.setNum(total_epoch)
        self.LB_TRAINING_EPOCH_LOSS.setNum(train_loss)
        self.LB_TRAINING_EPOCH_ACCURACY.setNum(train_accuracy)
        self.LB_VALIDATION_EPOCH_LOSS.setNum(validation_loss)
        self.LB_VALIDATION_EPOCH_ACCURACY.setNum(validation_accuracy)
        self.LB_TRAINING_IDX_CURRENT.setNum(train_idx)
        self.LB_TRAINING_IDX_TOTAL.setNum(train_total_idx)
        self.LB_STATE.setText(self.modelmanagement.state)
        self.timer = threading.Timer(0.5, self.timeout)
        self.timer.start()

    def on_thread(self):
        self.timer = threading.Timer(1, self.timeout)
        self.timer.start()
        self.modelmanagement.train()
        self.LB_STATE.setText(self.modelmanagement.state)
        time.sleep(1)
        self.timer.cancel()
        self.BTN_SAVE_PARAMETER.setEnabled(True)
        self.BTN_INFERENCE.setEnabled(True)

    def train(self):
        t = threading.Thread(target=self.on_thread)
        t.start()

    def set_optimizer_sgd(self):
        self.modelmanagement.set_optimizer('sgd')
        self.b_optimizer = True
        self.LB_STATE.setText(self.modelmanagement.state)
        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def set_optimizer_adam(self):
        self.modelmanagement.set_optimizer('adam')
        self.b_optimizer = True
        self.LB_STATE.setText(self.modelmanagement.state)

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def set_augmentation_radiobutton(self):
        if self.RB_DATA_AUGMENTATION_YES.isChecked():
            self.RB_DATA_AUGMENTATION_NO.setChecked(False)
        elif self.RB_DATA_AUGMENTATION_NO.isChecked():
            self.RB_DATA_AUGMENTATION_YES.setChecked(False)

    def save_training_parameter(self):
        learning_rate = float(self.LE_PARAMETER_LEARNING_RATE.text())
        epoch = int(self.LE_PARAMETER_EPOCH.text())
        batch_size = int(self.LE_PARAMETER_BATCH_SIZE.text())
        input_shape = int(self.LE_PARAMETER_INPUT_SHAPE.text())

        if self.RB_DATA_AUGMENTATION_YES.isChecked():
            augment = True
        elif self.RB_DATA_AUGMENTATION_NO.isChecked():
            augment = False
        if self.CB_AUGMENT_ASPECT.isChecked():
            aspect = True
        else:
            aspect = False
        if self.CB_AUGMENT_FLIP.isChecked():
            flip = True
        else:
            flip = False
        if self.CB_AUGMENT_HSV.isChecked():
            hsv = True
        else:
            hsv = False
        if self.CB_AUGMENT_PCA.isChecked():
            pca = True
        else:
            pca = False

        if self.modelmanagement.image_net_train is not None:
            self.modelmanagement.set_training_parameter(learning_rate=learning_rate, epoch=epoch, ext='h5',
                                                    batch_size=batch_size, input_shape=input_shape,
                                                    augment=augment, aspect=aspect, flip=flip, hsv=hsv, pca=pca)
        if self.modelmanagement.image_net_validation is not None:
            self.modelmanagement.set_validation_parameter(input_shape=input_shape)
        if self.modelmanagement.image_net_test is not None:
            self.modelmanagement.set_testing_parameter(input_shape=input_shape)
        self.BTN_RESNET34.setEnabled(True)
        self.BTN_RESNET18.setEnabled(True)
        self.BTN_RESNET50.setEnabled(True)
        self.BTN_RESNET101.setEnabled(True)
        self.BTN_RESNET152.setEnabled(True)
        self.BTN_VGG16.setEnabled(True)
        self.BTN_VGG19.setEnabled(True)
        self.BTN_MOBILENET_V1.setEnabled(True)
        self.BTN_MOBILENET_V2.setEnabled(True)
        self.BTN_DENSENET201.setEnabled(True)
        self.BTN_DENSENET121.setEnabled(True)
        self.BTN_DENSENET169.setEnabled(True)

    def load_vggnet16(self):
        self.modelmanagement.load_model('vggnet_16')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_vggnet19(self):
        self.modelmanagement.load_model('vggnet_19')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)


    def load_resnet18(self):
        self.modelmanagement.load_model('resnet_18')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_resnet34(self):
        self.modelmanagement.load_model('resnet_34')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_resnet50(self):
        self.modelmanagement.load_model('resnet_50')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_resnet101(self):
        self.modelmanagement.load_model('resnet_101')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_resnet152(self):
        self.modelmanagement.load_model('resnet_152')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_mobilenet_v1(self):
        self.modelmanagement.load_model('mobilenet_v1')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_mobilenet_v2(self):
        self.modelmanagement.load_model('mobilenet_v2')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_densenet121(self):
        self.modelmanagement.load_model('densenet_121')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_densenet169(self):
        self.modelmanagement.load_model('densenet_169')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_densenet201(self):
        self.modelmanagement.load_model('densenet_201')
        self.LB_STATE.setText(self.modelmanagement.state)
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(self.modelmanagement.summary)
        self.BTN_TRAIN.setEnabled(True)
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def load_testing_set(self):
        data_path = QFileDialog.getExistingDirectory(self.qwidget, "Select Directory")
        self.modelmanagement.load_test_dataset(data_path)
        self.LB_STATE.setText(self.modelmanagement.state)
        self.LB_TESTING_DATA_SET_SIZE.setNum(self.modelmanagement.test_data_size)

    def load_validation_set(self):
        data_path = QFileDialog.getExistingDirectory(self.qwidget, "Select Directory")
        label_file, _ = QFileDialog.getOpenFileName(self.qwidget, 'Open Label file', "",
                                                    "All Files(*);; Python Files(*.txt)", '/')
        self.modelmanagement.load_validation_dataset(data_path, label_file)
        self.LB_STATE.setText(self.modelmanagement.state)
        self.LB_VALIDATION_DATA_SET_SIZE.setNum(self.modelmanagement.validation_data_size)

        if self.modelmanagement.validation_data_size > 0 or self.modelmanagement.train_data_size > 0:
            train_rate, validation_rate = self.modelmanagement.get_train_validation_rate()
            self.LB_RATE_TRAINING.setNum(train_rate)
            self.LB_RATE_VALIDATION.setNum(validation_rate)
            self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_training_set(self):
        data_path = QFileDialog.getExistingDirectory(self.qwidget, "Select Directory")
        label_file, _ = QFileDialog.getOpenFileName(self.qwidget, 'Open Label file', "",
                                                    "All Files(*);; Python Files(*.txt)", '/')
        self.modelmanagement.load_train_dataset(data_path, label_file)
        self.LB_STATE.setText(self.modelmanagement.state)
        self.LB_TRAINING_DATA_SET_SIZE.setNum(self.modelmanagement.train_data_size)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QMainWindow()
    ui = MainWindow(w)
    w.show()
    sys.exit(app.exec_())

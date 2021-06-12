import tensorflow as tf
import os

from DataManagement.Util.augmentation import *



class DataManagement:

    def __init__(self, data_path, list_file=None, classes=1000, mode=0):
        # mode : 0 -> train, 1 -> validation, 2 -> test

        self.mode = mode
        self.list_file = list_file
        self.data_path = data_path
        self.images = []
        self.labels = []
        self.dataset = 0
        self.classes = classes

        self.b_augment = False
        self.b_augment_aspect = False
        self.b_augment_flip = False
        self.b_augment_hsv = False
        self.b_augment_pca = False

        self.input_shape = (224, 224, 3)    # Default Setting
        self.batch_size = 2                 # Default Setting

        self.load_data()

    def load_list(self):
        images = []
        labels = []
        if self.mode is 0 or self.mode is 1:
            with open(self.list_file, 'r') as f:
                for line in f:
                    line = line.replace('\n', '').split(' ')
                    if os.path.isfile(os.path.join(self.data_path, line[0])):  # 파일의 존재유무 확인
                        images.append(os.path.join(self.data_path, line[0]))
                        labels.append(int(line[1]))
        else:
            for (path, dir, files) in os.walk(self.data_path):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext == '.JPEG' or ext == '.png' or ext == '.jpg':
                        images.append(path + '/' + filename)
                        labels.append(-1)
        return images, labels

    # R-IR-SFR-012
    def load_data(self):
        self.images, self.labels = self.load_list()
        self.dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        if self.mode is 0:
            self.dataset = self.dataset.shuffle(len(self.images))
            self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(
            lambda x, y: tf.py_function(self.load_image, inp=[x, y], Tout=[tf.float32, tf.float32, tf.string]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def get_data_size(self):
        return len(self.images)

    def set_batch_size(self, batch_size=2):
        self.batch_size = batch_size
        self.dataset = self.dataset.batch(batch_size)

    # R-IR-SFR-014
    def set_augment(self, augment, aspect, flip, hsv, pca):
        self.b_augment = augment
        self.b_augment_aspect = aspect
        self.b_augment_flip = flip
        self.b_augment_hsv = hsv
        self.b_augment_pca = pca

    def set_input_shape(self, size):
        self.input_shape = (size, size, 3)

    def get_data(self):
        return self.dataset.__iter__()

    # R-IR-SFR-013
    def load_image(self, image_path, label):
        image = cv2.imread(image_path.numpy().decode()).astype(np.float32)

        if self.b_augment:
            if self.b_augment_aspect:
                image = random_aspect(image)

            image = random_size(image)
            image = random_crop(image, self.input_shape)

            if self.b_augment_flip:
                image = random_flip(image)
            if self.b_augment_hsv:
                image = random_hsv(image)
            if self.b_augment_pca:
                image = random_pca(image)
        else:
            image = random_size(image, target_size=256)
            image = center_crop(image, self.input_shape)

        image = normalize(image)

        label_one_hot = np.zeros(self.classes)
        label_one_hot[label] = 1.0

        return image, label_one_hot, image_path.numpy().decode()



'''
import tkinter
from tkinter import filedialog
import numpy
import cv2


# Test Code
root = tkinter.Tk()
root.withdraw()
label_file = filedialog.askopenfilename(initialdir="/", title='Please Select Label File',
                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
data_path = filedialog.askdirectory(parent=root, initialdir="/", title='Please Select Image Data Dir')
image_net_data = DataManagement(data_path, list_file=label_file)
image_net_data.set_augment(True, True, True, True, True)
image_net_data.set_batch_size(7)
image_net_data.set_input_shape(240)

images, labels = image_net_data.get_data().next()
# print(np.shape(images), np.shape(labels))
for i in range(images.shape[0]):
    cv2.imshow('show', images[i].numpy().astype(np.uint8))
    cv2.waitKey(0)

print('Get Data Size : ' + str(image_net_data.get_data_size()))

label_file = filedialog.askopenfilename(initialdir="/", title='Please Select Label File',
                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
data_path = filedialog.askdirectory(parent=root, initialdir="/", title='Please Select Image Data Dir')
image_net_validation_data = DataManagement(data_path, list_file=label_file, mode=1 )
image_net_validation_data.set_augment(False, False, False, False, False)
image_net_validation_data.set_input_shape(224)
image_net_validation_data.set_batch_size(3)

images, labels = image_net_data.get_data().next()
# print(np.shape(images), np.shape(labels))
for i in range(images.shape[0]):
    cv2.imshow('show', images[i].numpy().astype(np.uint8))
    cv2.waitKey(0)

print('Get Data Size : ' + str(image_net_validation_data.get_data_size()))
'''

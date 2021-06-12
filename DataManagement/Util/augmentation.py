import cv2
import numpy as np

"""
|------|
|      | height, y
|      |
|------|
 width, x
"""


def random_size(image, target_size=None):                       # 가로/세로 크기 조절
    height, width, _ = np.shape(image)
    if target_size is None:
        short_side_scale = (256, 384)
        target_size = np.random.randint(*short_side_scale)
    if height < width:
        size_ratio = target_size / height
    else:
        size_ratio = target_size / width
    resize_shape = (
    int(width * size_ratio), int(height * size_ratio))
    return cv2.resize(image, resize_shape)


def random_aspect(image):                                       # 가로/세로 비율
    height, width, _ = np.shape(image)
    aspect_ratio_scale = (0.8, 1.25)
    aspect_ratio = np.random.uniform(*aspect_ratio_scale)
    if height < width:
        resize_shape = (int(width * aspect_ratio), height)
    else:
        resize_shape = (width, int(height * aspect_ratio))
    return cv2.resize(image, resize_shape)


def random_crop(image, input_shape):                            # 가로/세로 잘라내기
    height, width, _ = np.shape(image)
    input_height, input_width, _ = input_shape
    crop_x = np.random.randint(0, width - input_width)
    crop_y = np.random.randint(0, height - input_height)
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]


def random_flip(image):                                         # 좌우반전
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    return image


def random_hsv(image):                                          # HSV 변환
    hue_delta = (-36, 36)
    saturation_scale = (0.6, 1.4)
    brightness_scale = (0.6, 1.4)

    random_h = np.random.uniform(*hue_delta)
    random_s = np.random.uniform(*saturation_scale)
    random_v = np.random.uniform(*brightness_scale)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0] = image_hsv[:, :, 0] + random_h % 360.0              # hue
    image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * random_s, 1.0)     # saturation
    image_hsv[:, :, 2] = np.minimum(image_hsv[:, :, 2] * random_v, 255.0)   # brightness

    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)


def random_pca(image):                                          # PCA 변환
    pca_std = 0.1
    eigval = [55.46, 4.794, 1.148]
    eigvec = [[-0.5836, -0.6948, 0.4203],
              [-0.5808, -0.0045, -0.8140],
              [-0.5675, 0.7192, 0.4009]]

    alpha = np.random.normal(0, pca_std, size=(3,))
    offset = np.dot(eigvec * alpha, eigval)
    image = image + offset
    return np.maximum(np.minimum(image, 255.0), 0.0)


def normalize(image):                                           # Normalize
    mean = [103.939, 116.779, 123.68]
    std = [58.393, 57.12, 57.375]

    for i in range(3):
        image[..., i] = (image[..., i] - mean[i]) / std[i]
    return image


def center_crop(image, input_shape):
    height, width, _ = np.shape(image)
    input_height, input_width, _ = input_shape
    crop_x = (width - input_width) // 2
    crop_y = (height - input_height) // 2
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

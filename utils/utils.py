# coding:utf8
import cv2
import numpy as np
from model.facenet import get_model
from keras.models import Model


def get_feature_model():
    """Get face features extraction model.

    # Returns
        feat_model: Model, face features extraction model.
    """
    model = get_model((64, 64, 3))
    model.load_weights('model/weight.h5')

    feat_model = Model(inputs=model.get_layer('model_1').get_input_at(0),
                       outputs=model.get_layer('model_1').get_output_at(0))

    return feat_model


def process_image(img):
    """Resize, reduce and expand image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (64, 64),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

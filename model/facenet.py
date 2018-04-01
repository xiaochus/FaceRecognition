"""Face features model.
This network make similar face features closer.
"""

from .mobilenet_v2 import MobileNetv2

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.utils.vis_utils import plot_model


def euclidean_distance(inputs):
    """Euclidean Distance
    This function calculate the euclidean distance of two features.

    # Arguments
        inputs: List, two features.
    # Returns
        Output: Double, euclidean distance.
    """
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    """Contrastive Loss
    This function calculate the contrastive loss.

    # Arguments
        y_true: Integer, Pair mark.
        y_pred: Double, Euclidean Distance.
    # Returns
        Output: Double, contrastive loss.
    """
    margin = 1.
    """
    l1 = K.mean(y_true) * K.square(y_pred)
    l2 = (1 - y_true) *  K.square(K.maximum(margin - y_pred, 0.))
    loss = l1 + l2
    """
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))


def get_model(shape):
    """Face features network
    This network make similar face features closer.

    # Arguments
        shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
    # Returns
        Output Model.
    """
    mn = MobileNetv2(shape)

    im1 = Input(shape=shape)
    im2 = Input(shape=shape)

    feat1 = mn(im1)
    feat2 = mn(im2)

    distance = Lambda(euclidean_distance)([feat1, feat2])

    face_net = Model(inputs=[im1, im2], outputs=distance)
    face_net.compile(optimizer='adam', loss=contrastive_loss)
    plot_model(face_net, to_file='images/face_net.png', show_shapes=True)

    return face_net

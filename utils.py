# coding:utf8
import cv2
from model.facenet import get_model
from keras.models import Model


def detect_face(frame):
    pic = frame.copy()
    cpath = 'data/haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cpath)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        pic = cv2.rectangle(pic, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return pic


def get_feature_model():
    model = get_model((64, 64, 3))
    model.load_weights('model/weight.h5')

    feat_model = Model(inputs=model.get_layer('model_1').get_input_at(0),
                       outputs=model.get_layer('model_1').get_output_at(0))

    return feat_model

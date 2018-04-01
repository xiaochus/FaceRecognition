"""Face detection model.
"""

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, type, threshold=0.5):
        """Init.
        """
        self.type = type
        self.t = threshold

        if type == 'harr':
            self.detector = self._create_harr_detector()
        elif type == 'ssd':
            self.detector = self._create_ssd_detector()
        else:
            raise 'You must select a FaceDetector type!'

    def _create_haar_detector(self):
        """Create haar cascade classifier.

        # Arguments
            path: String, path to xml data.

        # Returns
            face_cascade: haar cascade classifier.
        """
        path = 'data/haarcascades/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(path)

        return face_cascade

    def _create_ssd_detector(self):
        """Create ssd face classifier.

        # Returns
            ssd: ssd 300 * 300 face classifier.
        """
        prototxt = 'data/ssd/deploy.prototxt.txt'
        model = 'data/ssd/ssd300.caffemodel'
        ssd = cv2.dnn.readNetFromCaffe(prototxt, model)

        return ssd

    def _ssd_box(self, detections, h, w):
        """Resize the detection boxes of ssd.

        # Arguments
            detections: String, path to xml data.
            h: Integer, original height of frame.
            w: Integer, original width of frame.

        # Returns
            rects: detection boxes.
        """
        rects = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < self.t:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            rects.append((x1, y1, x2 - x1, y2 - y1))

        return rects

    def detect(self, frame):
        """Detect face with haar cascade classifier.

        # Arguments
            frame: ndarray(n, n, 3), video frame.

        # Returns
            faces: List, faces rectangles in the frame.
        """
        pic = frame.copy()

        if self.type == 'harr':
            gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)
        if self.type == 'ssd':
            h, w = pic.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(pic, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            self.detector.setInput(blob)
            detections = self.detector.forward()
            faces = self._ssd_box(detections, h, w)

        return faces

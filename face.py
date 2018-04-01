"""Face recognition of PC camera.
"""

import os
import cv2
import numpy as np
import utils.utils as u
from utils.window_manager import WindowManager
from utils.face_detector import FaceDetector


class Face:
    def __init__(self, threshold):
        """Init.

        # Arguments
            threshold: Float, threshold for specific face.
        """
        self._t = threshold
        self._key = self._load_key()
        self._key_cache = []
        self._model = u.get_feature_model()
        self._windowManager = WindowManager('Face', self.on_keypress)
        self._faceDetector = FaceDetector('ssd', 0.5)

    def run(self):
        """Run the main loop.
        """
        capture = cv2.VideoCapture(0)

        self._windowManager.create_window()
        while self._windowManager.is_window_created:

            success = capture.grab()
            _, frame = capture.retrieve()

            if frame is not None and success:
                faces = self._faceDetector.detect(frame)

                if self._key is not None and faces is not None:
                    label = self._compare_distance(frame, faces)
                    f = self._draw(frame, faces, label)
                else:
                    f = self._draw(frame, faces)

                self._windowManager.show(f)
            self._windowManager.process_events(frame, faces)

    def _load_key(self):
        """Load the key feature.
        """

        kpath = 'data/key.npy'

        if os.path.exists(kpath):
            key = np.load('data/key.npy')
        else:
            key = None

        return key

    def _get_feat(self, frame, face):
        """Get face feature from frame.

        # Arguments
            frame: ndarray, video frame.
            face: tuple, coordinates of face in the frame.

        # Returns
            feat: ndarray (128, ), face feature.
        """
        x, y, w, h = face
        img = frame[y: y + h, x: x + w, :]
        image = u.process_image(img)
        feat = self._model.predict(image)[0]

        return feat

    def _compare_distance(self, frame, faces):
        """Compare faces feature in the frame with key.

        # Arguments
            frame: ndarray, video frame.
            faces: List, coordinates of faces in the frame.

        # Returns
            label: list, if match the key.
        """
        label = []

        for (x, y, w, h) in faces:
            feat = self._get_feat(frame, (x, y, w, h))

            dist = []
            for k in self._key:
                dist.append(np.linalg.norm(k - feat))
            dist = min(dist)
            print(dist)
            if dist < self._t:
                label.append(1)
            else:
                label.append(0)
        print(label)
        return label

    def _draw(self, frame, faces, label=None):
        """Draw the rectangles in the frame.

        # Arguments
            frame: ndarray, video frame.
            faces: List, coordinates of faces in the frame.
            label: List, if match the key.

        # Returns
            f: ndarray, frame with rectangles.
        """
        f = frame.copy()
        color = [(0, 0, 255), (255, 0, 0)]
        if label is None:
            label = [0 for _ in range(len(faces))]

        for rect, i in zip(faces, label):
            (x, y, w, h) = rect
            f = cv2.rectangle(f, (x, y),
                              (x + w, y + h),
                              color[i], 2)

        return f

    def on_keypress(self, keycode, frame, faces):
        """Handle a keypress event.
        Press esc to  quit window.
        Press space 5 times to record different gestures of the face.

        # Arguments
            keycode: Integer, keypress event.
            frame: ndarray, video frame.
            faces: List, coordinates of faces in the frame.
        """
        if keycode == 32:  # space -> save face id.
            nums = len(self._key_cache)

            if nums < 5:
                feat = self._get_feat(frame, faces[0])
                self._key_cache.append(feat)
                print('Face id {0} recorded!'.format(nums + 1))
            else:
                np.save('data/key.npy', np.array(self._key_cache))
                print('All face ID recorded!')
                self._key = self._key_cache
                self._key_cache = []
        elif keycode == 27:  # escape -> quit
            self._windowManager.destroy_window()


if __name__ == '__main__':
    face = Face(0.3)
    face.run()

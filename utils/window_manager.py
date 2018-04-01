"""
Window Manager.
"""

import cv2


class WindowManager:
    def __init__(self, windowname, keypresscallback=None):
        """Init.

        # Arguments
            windowname: String, name of window.
            keypresscallback: function, key process function.
        """
        self.keypressCallback = keypresscallback
        self._windowName = windowname
        self._isWindowCreated = False

    @property
    def is_window_created(self):
        """If the window is created.
        """
        return self._isWindowCreated

    def create_window(self):
        """Create a video window.
        """
        cv2.namedWindow(self._windowName, cv2.WINDOW_NORMAL)
        self._isWindowCreated = True

    def show(self, frame):
        """show the frame in the window.
        """
        cv2.imshow(self._windowName, frame)

    def destroy_window(self):
        """Destroy the video window.
        """
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def process_events(self, frame, faces):
        """Process the key event.
        """
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode, frame, faces)

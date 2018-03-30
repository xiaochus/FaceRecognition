"""Data process.
Data process and generation.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def read_img(path):
    """Read image
    This function read images from folders for different person.

    # Arguments
        path: String, path of database.
    # Returns
        res: List, images for different person.
    """
    res = []

    for (root, dirs, files) in os.walk(path):
        if files:
            tmp = []
            for f in files[-4:]:
                img = os.path.join(root, f)
                image = cv2.imread(img)
                image = cv2.resize(image, (64, 64),
                                   interpolation=cv2.INTER_CUBIC)
                image = np.array(image, dtype='float32')
                image /= 255.
                tmp.append(image)

            res.append(tmp)

    return res


def get_paris(path):
    """Make pairs.
    This function make pairs for same person and different person.

    # Arguments
        path: String, path of database.
    # Returns
        sm1: List, first object in pairs.
        sm2: List, second object in pairs.
        y1: List, pairs mark (same: 0, different: 1).
    """
    sm1, sm2, df1, df2 = [], [], [], []
    res = read_img(path)

    persons = len(res)

    for i in range(persons):
        for j in range(i, persons):
            p1 = res[i]
            p2 = res[j]

            if i == j:
                for pi in p1:
                    for pj in p2:
                        sm1.append(pi)
                        sm2.append(pj)
            else:
                df1.extend(p1)
                df2.extend(p2)

    df1 = df1[:len(sm1)]
    df2 = df2[:len(sm2)]
    y1 = list(np.zeros(len(sm1)))
    y2 = list(np.ones(len(df1)))

    sm1.extend(df1)
    sm2.extend(df2)
    y1.extend(y2)

    return sm1, sm2, y1


def create_generator(x, y, batch):
    """Create data generator.
    This function is a data generator.

    # Arguments
        x: List, Input data.
        y: List, Data label.
        batch: Integer, batch size for data generator.
    # Returns
        [x1, x2]: List, pairs data with batch size.
        yb: List, Data label.
    """
    while True:
        index = np.random.choice(len(y), batch)
        x1, x2, yb = [], [], []
        for i in index:
            x1.append(x[i][0])
            x2.append(x[i][1])
            yb.append(y[i])
        x1 = np.array(x1)
        x2 = np.array(x2)

        yield [x1, x2], yb


def get_train_test(path):
    """Get train and test data
    This function split train and test data and shuffle it.

    # Arguments
        path: String, path of database.
    # Returns
        X_train: List, Input data for train.
        X_test: List, Data label for train.
        y_train: List, Input data for test.
        y_test: List, Data label for test.
    """
    im1, im2, y = get_paris(path)
    im = list(zip(im1, im2))

    X_train, X_test, y_train, y_test = train_test_split(
        im, y, test_size=0.33)

    return X_train, X_test, y_train, y_test

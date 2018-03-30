"""Visualization of experiment results
"""
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import get_feature_model


def plot_loss():
    """Plot loss and val loss.
    """
    df = pd.read_csv('data/loss.csv', encoding='utf-8')
    loss = df['loss'].values
    val_loss = df['val_loss'].values
    x = [i for i in range(1, len(loss) + 1)]

    plt.plot(x, loss, label='Train loss')
    plt.plot(x, val_loss, label='Val loss')

    plt.xlabel('Epochs')
    plt.ylabel('Contrastive Loss')
    plt.title('Train and test loss')
    plt.grid(True)
    plt.legend(shadow=True, fontsize='x-large')

    plt.show()


def plot_reduce_dimension():
    """Plot reduced dimension result wiht t-SNE.
    """
    model = get_feature_model()

    outputs = []
    n = 8
    paths = 'data/val'

    for (root, dirs, files) in os.walk(paths):
        if files:
            for f in files:
                img = os.path.join(root, f)
                image = cv2.imread(img)
                image = cv2.resize(image, (64, 64),
                                   interpolation=cv2.INTER_CUBIC)
                image = np.array(image, dtype='float32')
                image /= 255.
                image = np.expand_dims(image, axis=0)

                output = model.predict(image)[0]
                outputs.append(output)

    embedded = TSNE(2).fit_transform(outputs)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i in range(n):
        m, n = i * 20, (i + 1) * 20
        plt.scatter(embedded[m: n, 0], embedded[m: n, 1],
                    c=colors[i], alpha=0.5)

    plt.title('T-SNE')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_loss()
    plot_reduce_dimension()

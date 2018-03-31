"""Visualization of experiment results
"""
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import get_feature_model, process_image


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


def plot_reduce_dimension(model):
    """Plot reduced dimension result wiht t-SNE.

    # Arguments
        model: Model, face features extraction model.
    """

    outputs = []
    n = 8
    paths = 'data/grimace'
    dirs = np.random.choice(os.listdir(paths), n)

    for d in dirs:
        p = paths + '/' + str(d)
        files = os.listdir(p)
        if files:
            for f in files:
                img = os.path.join(p, f)
                image = cv2.imread(img)
                image = process_image(image)
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


def compare_distance(model):
    """Compare the features distances of different people.

    # Arguments
        model: Model, face features extraction model.
    """

    dists = []
    outputs = []
    paths = 'images/person/'

    for i in range(6):
        img = paths + str(i) + '.jpg'
        image = cv2.imread(img)
        image = process_image(image)

        output = model.predict(image)[0]
        outputs.append(output)

    vec1 = outputs[0]
    for vec2 in outputs:
        dist = np.linalg.norm(vec1 - vec2)
        dists.append(dist)

    print(dists[1:])

    plt.bar(range(1, 6), (dists[1:]), color='lightblue')
    plt.xlabel('Person')
    plt.ylabel('Euclidean distance')
    plt.title('Similarity')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    model = get_feature_model()

    plot_loss()
    plot_reduce_dimension(model)
    compare_distance(model)

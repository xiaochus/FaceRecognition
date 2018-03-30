"""Train the face recognition network.
"""
import pandas as pd
from model.facenet import get_model
from data.generator import get_train_test, create_generator


def train():
    """Train the face recognition network.
    """
    model = get_model((64, 64, 3))
    X_train, X_test, y_train, y_test = get_train_test('data/faces96')

    hist = model.fit_generator(create_generator(X_train, y_train, 40),
                               steps_per_epoch=70,
                               epochs=50,
                               validation_data=create_generator(X_test, y_test, 40),
                               validation_steps=40)

    model.save_weights('model/weight.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('data/loss.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    train()

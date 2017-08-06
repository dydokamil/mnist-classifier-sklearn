import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier


def shift_image(image, direction):
    if direction not in ['right', 'left', 'up', 'down']:
        raise ValueError()

    image = reshape_image(image)
    if direction == 'down':
        img_added_col = np.vstack((np.zeros(image.shape[1]), image))
        return np.asarray(img_added_col[:28, :]).flatten()
    elif direction == 'up':
        img_added_col = np.vstack((image, np.zeros(image.shape[1])))
        return np.asarray(img_added_col[1:, :]).flatten()
    elif direction == 'left':
        img_added_col = np.hstack((image, np.zeros(image.shape[0]).reshape((-1, 1))))
        return np.asarray(img_added_col[:, 1:]).flatten()
    elif direction == 'right':
        img_added_col = np.hstack((np.zeros(image.shape[0]).reshape((-1, 1)), image))
        return np.asarray(img_added_col[:, :28]).flatten()


def reshape_image(image):
    shape_sqrt = int(np.sqrt(*image.shape))
    return image.reshape((shape_sqrt, shape_sqrt))


def show_image(image):
    if len(image.shape) == 1:
        image = reshape_image(image)
    plt.imshow(image)


mnist = fetch_mldata('MNIST original')
X = mnist['data']
y = mnist['target']

# show_image(X[0])
# shifted = shift_image(X[0], 'left')
# show_image(shifted)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
X_train_shifted = np.zeros((X_train.shape[0] * 5, X_train.shape[1]))

# data augmentation
for idx, image in enumerate(X_train):
    shifted_right = shift_image(image, 'right')
    shifted_left = shift_image(image, 'left')
    shifted_up = shift_image(image, 'up')
    shifted_down = shift_image(image, 'down')
    X_train_shifted[idx * 5] = image
    X_train_shifted[idx * 5 + 1] = shifted_right
    X_train_shifted[idx * 5 + 2] = shifted_down
    X_train_shifted[idx * 5 + 3] = shifted_left
    X_train_shifted[idx * 5 + 4] = shifted_up

X_train = X_train_shifted

# fix labels
y_train = y_train.repeat(5)

shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

clf = RandomForestClassifier(n_estimators=250, max_features='auto', n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

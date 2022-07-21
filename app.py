from mnist_digits_keras import start_keras_app
from mnist_digits_sklearn import start_sklearn_app

import matplotlib.pyplot as plt

from keras.datasets import mnist



# Entry point for mnist_digits_sklearn.__init__.py
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_classes = 10
    f, ax = plt.subplots(1, num_classes, figsize=(20,20))
    for i in range(0, num_classes):
        sample = x_train[y_train == i][0]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title("Label: {}".format(i), fontsize=16)
        ax[i].axis('off')
    plt.show()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    start_sklearn_app(x_train, y_train, x_test, y_test)
    start_keras_app(x_train, y_train, x_test, y_test)